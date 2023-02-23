import getpass
from collections import Counter
import random

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torch import from_numpy
from rampwf.score_types.base import BaseScoreType
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
    ToTensor(),
])


class ImageSet(Dataset):
    def __init__(self, paths, transform, preload=False):
        self.paths = paths
        self.transform = transform
        self.preload = preload
        if self.preload:
            self.files = [
                self.transform(
                    Image.open(path)
                ) for path in self.paths]

    def __getitem__(self, index):
        if self.preload:
            return self.files[index]
        else:
            return self.transform(
                Image.open(self.paths[index])
            )

    def __len__(self):
        return len(self.paths)


class Master():
    def __init__(self, n_fold=3):
        self.batch_size = 64
        self.score = {}
        # [None, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        self.pattern = [None] + [i for i in range(n_fold) for z in range(3)] + 50 * [n_fold]  # 6
        self.memory_call = Counter()
        self.memory = Counter()
        self.n_fold = n_fold

    def eval(self, y_true, y_pred, metric):
        assert metric in ("FID", "KID_mean", "KID_std")
        self.memory_call[metric] += 1
        current_fold: int = self.pattern[self.memory_call[metric]]  # retrieve position in k_fold
        context = (metric, current_fold)
        self.memory[context] += 1  # we count th enumber of call of each metric

        if context in self.score:
            return self.score[context]

        if current_fold == self.n_fold:
            # Bagged score
            print("bagged score")
            return 1.

        if len(y_true) == 0:
            # assert self.memory[metric] == 3
            # print(f"len(y_true) == 0 and {self.memory[context]=}")
            return self.score[context]

        if metric == "FID":
            metric_ = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        elif "KID" in metric:
            metric_ = KernelInceptionDistance(reset_real_features=True, normalize=True).to(device)

        i = -1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch / 255).to(device)
            metric_.update(batch_, real=False)

        if i == -1:
            # assert self.memory[metric] == 2
            # print(f"i==-1 and {self.memory[context]=}")
            return self.score[context]

        # Handling true data
        folders = set(path_.parent.name for path_ in y_true)
        assert len(folders) == 3
        y_true_ = tuple(path for path in y_true if path.parent.name == f"train_{current_fold+1}")

        if getpass.getuser() == "Alexandre":  # For Alexandre, we use much less data becausee his computer is in parper
            s_y_true = list(y_true_)
            random.shuffle(s_y_true)
            dataset = ImageSet(
                paths=s_y_true[:50],
                transform=transform,
                preload=True,
            )
        else:
            dataset = ImageSet(
                paths=y_true_,
                transform=transform,
                preload=True,
            )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False  # TODO
        )
        for batch in loader:
            batch_ = batch.to(device)
            metric_.update(batch_,
                       real=True)  # RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 536406000 bytes.

        # Compute score
        if metric == "FID":
            self.score[context] = metric_.compute().item()

            # Destroy the former instance of FID
            # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
            return self.score[context]
        elif "KID" in metric:
            kid_mean, kid_std = metric_.compute()
            self.score[("KID_mean", current_fold)] = kid_mean.item()
            self.score[("KID_std", current_fold)] = kid_std.item()
            return self.score[context]




MASTER = Master()


class FID(BaseScoreType):
    precision = 1

    def __init__(self, name="FID"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="FID")


# Kernel Inception Distance (KID)

class KIDMean(BaseScoreType):
    precision = 1

    def __init__(self, name="KID_mean"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="KID_mean")

class KIDStd(BaseScoreType):
    precision = 1

    def __init__(self, name="KID_mean"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="KID_std")


# Inception Score

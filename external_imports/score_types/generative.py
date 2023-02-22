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


class FID_master():
    def __init__(self, n_fold=3):
        self.batch_size = 500
        self.score = {}
        # [None, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        self.pattern = [None] + [i for i in range(n_fold) for z in range(3)] + 50 * [n_fold]  # 6
        self.memory_call = Counter()
        self.memory = Counter()
        self.n_fold = n_fold

    def eval(self, y_true, y_pred, metric):
        assert metric == "FID"
        self.memory_call[metric] += 1
        current_fold: int = self.pattern[self.memory_call[metric]]  # retrieve position in k_fold
        context = (metric, current_fold)
        self.memory[context] += 1  # we count th enumber of call of each metric
        if current_fold == self.n_fold:
            # Bagged score
            print("bagged score")
            return 1.

        if len(y_true) == 0:
            # assert self.memory[metric] == 3
            # print(f"len(y_true) == 0 and {self.memory[context]=}")
            return self.score[context]

        fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)

        i = -1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch / 255).to(device)
            fid.update(batch_, real=False)

        if i == -1:
            # assert self.memory[metric] == 2
            # print(f"i==-1 and {self.memory[context]=}")
            return self.score[context]

        # Handling true data
        if getpass.getuser() == "Alexandre":  # For Alexandre, we use much less data becausee his computer is in parper
            s_y_true = list(y_true)
            random.shuffle(s_y_true)
            dataset = ImageSet(
                paths=s_y_true[:50],
                transform=transform,
                preload=True,
            )
        else:
            dataset = ImageSet(
                paths=y_true,
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
            fid.update(batch_,
                       real=True)  # RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 536406000 bytes.

        # Compute score
        self.score[context] = fid.compute().item()

        # Destroy the former instance of FID
        # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        return self.score[context]


FID_MASTER = FID_master()


class Troll(BaseScoreType):
    precision = 1

    def __init__(self, name="troll"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return FID_MASTER.eval(y_true, y_pred, metric="FID")

        if len(y_true) == 0:
            # case where we are testing the model
            return 1.
        if 1:
            # print(f"{y_true[:4]=}")
            folders = set(path_.parent.name for path_ in y_true)
            print(f"{folders=}, {len(y_true)=}")

        # HELPER
        i = -1
        for i, batch in enumerate(y_pred):
            assert isinstance(batch, np.ndarray)
            print(f"-> {batch.shape=}")
        print(f"-> len y_pred : {i + 1}")

        return 1.


# Fréchet Inception Distance (FID)
class FID(BaseScoreType):
    precision = 2

    def __init__(self, name='fid_score'):
        # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        self.name = name
        self.batch_size = 32
        self.score = None

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        # y_pred is generator with len
        fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)

        i = -1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch / 255).to(device)
            # print(batch_.size())
            fid.update(batch_, real=False)
        # If the generator is empty, it means that we already went through
        # this dataset.
        if i == -1 and not self.score is None:
            print("Generator is empty. We reuse the same previous score.")
            return self.score

        # Handling true data
        dataset = ImageSet(
            paths=y_true,
            transform=transform,
            preload=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        for batch in loader:
            batch_ = batch.to(device)
            # print(batch_)
            fid.update(batch_, real=True)
        # Compute score
        self.score = fid.compute().item()

        # Destroy the former instance of FID
        # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        return self.score


# Kernel Inception Distance (KID)

# TODO: see if we can combine both into one type of score because computing the core twice
# is expensive.

class KID(object):
    def __init__(self, name='kid_mean'):
        self.kid = KernelInceptionDistance(reset_real_features=True).to(device)
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        for batch in y_true:
            batch_ = torch.Tensor(batch_).to(device)
            self.kid.update(batch_, real=True)
        for batch in y_pred:
            batch_ = torch.Tensor(batch).to(device)
            self.kid.update(batch_, real=False)
        score = self.kid.compute()
        # score = (mean, std)
        return score[0]


class KID(object):
    def __init__(self, name='kid_mean'):
        self.kid = KernelInceptionDistance(reset_real_features=True).to(device)
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        for batch in y_true:
            batch_ = torch.Tensor(batch_).to(device)
            self.kid.update(batch_, real=True)
        for batch in y_pred:
            batch_ = torch.Tensor(batch).to(device)
            self.kid.update(batch_, real=False)
        score = self.kid.compute()
        # score = (mean, std)
        return score[0]

# Inception Score

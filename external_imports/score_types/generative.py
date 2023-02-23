import getpass
from collections import Counter
import random

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch import from_numpy
from rampwf.score_types.base import BaseScoreType
import numpy as np
from PIL import Image

import warnings

def disable_kid_warnings():
    import warnings
    warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system?
    warnings.filterwarnings('ignore')  # Ignore everything
    # ignore everything does not work: ignore specific messages, using regex
    warnings.filterwarnings(
        'ignore', '.*UserWarning: Metric `Kernel Inception Distance`*')

disable_kid_warnings()

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
        self.batch_size = 32
        self.score = {}
        # [None, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        self.pattern = [None] + [i for i in range(n_fold) for z in range(3)] + 50 * [n_fold]  # 6
        self.memory_call = Counter()
        self.memory = Counter()
        self.n_fold = n_fold

    def eval(self, y_true, y_pred, metric):
        assert metric in ("FID", "KID_mean", "KID_std", "IS_mean", "IS_std")
        self.memory_call[metric] += 1
        current_fold: int = self.pattern[self.memory_call[metric]]  # retrieve position in k_fold
        context = (metric, current_fold)
        self.memory[context] += 1  # we count the number of call of each metric

        if context in self.score:
            return self.score[context]

        if current_fold == self.n_fold:
            # TODO: Bagged score
            #print("bagged score")
            return 1.

        if len(y_true) == 0:
            # assert self.memory[metric] == 3
            # print(f"len(y_true) == 0 and {self.memory[context]=}")
            return self.score[context]

        fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        kid = KernelInceptionDistance(reset_real_features=True, normalize=True).to(device)
        is_ = InceptionScore(normalize=True).to(device)

        i = -1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch / 255).to(device)
            fid.update(batch_, real=False)
            kid.update(batch_, real=False)
            is_.update(batch_)

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
            shuffle=False
        )
        for batch in loader:
            batch_ = batch.to(device)
            fid.update(batch_, real=True)
            kid.update(batch_, real=True)

        fid_score= fid.compute().item()
        self.score[("FID", current_fold)] = fid_score

        kid_mean, kid_std = kid.compute()
        # We rescale the KID scores because otherwise they are too small and too close to 0.
        self.score[("KID_mean", current_fold)] = kid_mean.item()*1000
        self.score[("KID_std", current_fold)] = kid_std.item()*1000

        is_mean, is_std = is_.compute()
        self.score[("IS_mean", current_fold)] = is_mean.item()
        self.score[("IS_std", current_fold)] = is_std.item()

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

    def __init__(self, name="KID_std"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="KID_std")


# Inception Score

class ISMean(BaseScoreType):
    precision = 1

    def __init__(self, name="IS_mean"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="IS_mean")
    
class ISStd(BaseScoreType):
    precision = 1

    def __init__(self, name="IS_std"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="IS_std")
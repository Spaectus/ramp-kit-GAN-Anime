import os
from pathlib import Path
import itertools
import getpass

import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from prediction_types.generative import make_generative_img
from workflows.image_generative import ImageGenerative

problem_title = "GAN Anime"


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

workflow = ImageGenerative()

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
width = 64
height = 64
channels = 3
p = channels * height * width

BaseGenerative = make_generative_img(
    height=height, width=width, label_names=np.arange(p), channels=channels
)

# TODO: Redefine Predictions to fit the challenge.

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

# Fréchet Inception Distance (FID)
class FID(BaseScoreType):
    def __init__(self, name='fid_score'):
        self.fid = FrechetInceptionDistance(reset_real_features=True)
        self.name = name

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        self.fid.update(y_true, real=True)
        self.fid.update(y_pred, real=False)
        score = self.fid.compute()
        return score

# Kernel Inception Distance (KID)

# TODO: see if we can combine both into one type of score because computing the core twice
# is expensive.

class KIDMean(BaseScoreType):
    def __init__(self, name='kid_mean'):
        self.kid = KernelInceptionDistance(reset_real_features=True)
        self.name = name

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        self.kid.update(y_true, real=True)
        self.kid.update(y_pred, real=False)
        score = self.kid.compute()
        return score[0]

class KIDStd(BaseScoreType):
    def __init__(self, name='fid_var'):
        self.kid = KernelInceptionDistance(reset_real_features=True)
        self.name = name

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        self.kid.update(y_true, real=True)
        self.kid.update(y_pred, real=False)
        score = self.kid.compute()
        return score[1]


# Inception Score

score_types = [
    # Fréchet Inception Distance
    FID(),
    KIDStd()
]

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, str_:str):
    assert Path("data/images").exists(), f"Please download the data with `python download_data.py`"
    paths = tuple(Path("data/images").glob("*.jpg"))
    n_images = len(paths)
    assert n_images, f"No jpg images found in data/images"
    if 1:
        import torchvision
        assert width == height, f"This part of code can handle only square image"
        dataset = torchvision.datasets.ImageFolder(root="data",
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(width),
                                   torchvision.transforms.CenterCrop(width),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    else:
        dataset = tuple(zip([np.empty((channels, width, height))], range(1)))

    test = os.getenv("RAMP_TEST_MODE", 0)
    # for the "quick-test" mode, use less data
    if test:
        n_images = 1_000
    print(f"{n_images * 3 * width * height=}")
    result = np.empty((n_images, channels, width, height))
    # we extract the n_images first image of the torchvision.datasets.ImageFolder
    for i, (tensor_, _) in itertools.islice(enumerate(dataset), n_images):
        result[i] = tensor_
    return result

def get_train_data(path="."):
    return _read_data(path, "train")

def get_test_data(path="."):
    return _read_data(path, "test")

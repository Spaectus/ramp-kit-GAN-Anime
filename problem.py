import os
from pathlib import Path
import itertools
import getpass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

from sklearn.model_selection import StratifiedShuffleSplit
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torch import from_numpy

from prediction_types.generative import make_generative_img
from workflows.image_generative import ImageGenerative

from prediction_types.generative import make_generative_img

problem_title = "GAN Anime"

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

# n_images_generated : the number of images that we ask the ramp competitor to generate per fold
workflow = ImageGenerative(n_images_generated=3000, latent_space_dimension=1024)

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
width = 64
height = 64
channels = 3
p = channels * height * width

Predictions = make_generative_img(
    height=height, width=width, channels=channels
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Will be executed line 232 by https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/utils/submission.py the folowing code :
"""
predictions_train_train = problem.Predictions(
        y_pred=y_pred_train, fold_is=train_is)
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os

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


# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

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

        i=-1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch/255).to(device)
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

score_types = [
    # Fréchet Inception Distance
    FID(),
    # KID()
]


# ----------------------------------------------------------------------------
# Cross-validation scheme
# ----------------------------------------------------------------------------


def get_cv(X, y):
    """
    Specify a cross-validation scheme Specify a way to split the ‘train’ data into training and validation sets.
    This should be done by defining a get_cv() function that takes the feature and target data as parameters and returns
    indices that can be used to split the data. If you are using a function with a random element, e.g.,
    StratifiedShuffleSplit() from scikit-learn, it is important to set the random seed. This ensures that the train and
    valuidation data will be the same for all participants.

    :param X:
    :param y:
    :return:
    """
    size = len(y)  # == number of images normally
    # print(f"Pour get_cv, {len(y)=}, we choose {size=}")
    n_fold = 3
    # We will divided the dataset in n_fold equal set of images
    support = np.linspace(0, n_fold, endpoint=False, num=size, dtype=int)
    arange = np.arange(size)
    for i in range(n_fold):
        vec_bool = (support == i)  # vector of bool
        # we convert vector of bool to vector of indices for train_is and valid_is
        train_is = arange[vec_bool]
        valid_is = arange[vec_bool]  # train data and valid data are same in our case
        yield train_is, valid_is


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, str_: str):
    if 0:
        assert Path("data/images").exists(), f"Please download the data with `python download_data.py`"
        paths = tuple(Path("data/images").glob("*.jpg"))
        n_images = len(paths)
        assert n_images, f"No jpg images found in data/images"
        if 1:
            import torchvision
            assert width == height, f"This part of the code can handle only square images"
            dataset = torchvision.datasets.ImageFolder(root="data",
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.Resize(width),
                                                           torchvision.transforms.CenterCrop(width),
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5)),
                                                       ]))
        else:
            dataset = tuple(zip([np.empty((channels, width, height))], range(1)))

    test = os.getenv("RAMP_TEST_MODE", 0)
    # for the "quick-test" mode, use less data
    if test:
        n_images = 100  # TODO

    res = tuple()
    train_folders = tuple(Path("data").glob("train_*"))
    for train_folder in train_folders:
        res += tuple(train_folder.glob("*.jpg"))
    if test:
        return res[:100], res[:100]
    return res, res

    # print(f"{n_images * 3 * width * height=}")
    result = np.empty((n_images, channels, width, height))
    # we extract the n_images first image of the torchvision.datasets.ImageFolder
    for i, (tensor_, _) in itertools.islice(enumerate(dataset), n_images):
        result[i] = tensor_

    # we do not use y, so we set it as a empty vector BUT with correct shape !
    return result, np.empty_like(result)


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")

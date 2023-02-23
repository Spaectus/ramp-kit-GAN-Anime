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

from prediction_types.generative import make_generative_img
from workflows.image_generative import ImageGenerative

from prediction_types.generative import make_generative_img
from score_types.generative import KIDMean, KIDStd, FID, ISMean, ISStd

problem_title = "GAN Anime"

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

# n_images_generated : the number of images that we ask the ramp competitor to generate per fold
workflow = ImageGenerative(n_images_generated=3000, latent_space_dimension=1024, y_pred_batch_size=32,
                           chunk_size_feeder=64,
                           seed=23)

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

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


score_types = [
    FID(),
    KIDMean(),
    KIDStd(),
    ISMean(),
    ISStd()
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
    assert isinstance(X, tuple)
    assert isinstance(y, tuple)
    assert len(X) == len(y), f"{len(X)=}  and {len(y)=}"

    train_folders = tuple((Path("data")).glob("train_*"))
    folders_names = sorted([p.name for p in train_folders])
    map_ = {k: v for v, k in enumerate(folders_names)}  # {'train_1': 0, 'train_2': 1, 'train_3': 2}

    support = np.array([map_[path_img.parent.name] for path_img in X])
    arange = np.arange(len(y))
    for i_fold in range(len(folders_names)):
        vec_bool = (support == i_fold)
        # we convert vector of bool to vector of indices for train_is and valid_is
        train_is = arange[vec_bool]
        valid_is = arange[vec_bool]  # train data and valid data are same in our case
        yield train_is, valid_is

    """
    # We will divided the dataset in n_fold equal set of images
    support = np.linspace(0, n_fold, endpoint=False, num=size, dtype=int)
    arange = np.arange(size)
    for i in range(n_fold):
        vec_bool = (support == i)  # vector of bool
        # we convert vector of bool to vector of indices for train_is and valid_is
        train_is = arange[vec_bool]
        valid_is = arange[vec_bool]  # train data and valid data are same in our case
        yield train_is, valid_is
    """

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, str_: str):
    res = tuple()
    train_folders = tuple(Path("data").glob("train_*"))
    assert len(train_folders), f"Please dowload the data with python download_data.py"
    for train_folder in train_folders:
        res += tuple(train_folder.glob("*.jpg"))

    test = os.getenv("RAMP_TEST_MODE", 0)
    # for the "quick-test" mode, use less data
    if test:
        rng = np.random.RandomState(seed=0)
        selection = tuple(rng.choice(res, size=6000, replace=False))
        print(f"Warning : Can't get correctly 3 folds in --quick-test, not enough data !")
        return selection, selection
    return res, res


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return tuple(), tuple() #_read_data(path, "test")

import os
import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from prediction_types.generative import make_generative_img

problem_title = "GAN Anime"

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
width = 64
height = 64
channels = 3
p = channels * height * width

BaseGenerative = make_generative_img(
    height=height, width=width, label_names=np.arange(p)
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

import os
import pandas as pd
import numpy as np

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

from prediction_types.generative import make_generative

problem_title = "GAN Anime"

# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------
p = 64 * 64

BaseGenerative = make_generative(
    label_names=np.arange(p))

# TODO: Redefine Predictions to fit the challenge.

# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------
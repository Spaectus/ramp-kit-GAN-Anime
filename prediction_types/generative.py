"""Generation predictions.
y_pred can be one or two-dimensional (for multi-target regression)
"""

# Author: MAI Huu Tan

import numpy as np
from rampwf.prediction_types.base import BasePrediction

class BaseImgGen(BasePrediction):
    def valid_indexes(self):
        if len(self.y_pred.shape) <= 3:
            ValueError('y_pred.shape <= 3 is not implemented')
        elif len(self.y_pred.shape) == 4:
            # (n_samples, channel, height, width)
            return ~np.isnan(self.y_pred).any(axis=1)
        else:
            raise ValueError('y_pred.shape > 4 is not implemented')
        
    def check_y_pred_dimensions(self):
        # TODO: Implement this for images.
        pass


def _generation_init(self, y_pred=None, y_true=None, n_samples=None,
                     fold_is=None):
    """Initialize a generation prediction type.
    The input is either y_pred, or y_true, or n_samples.
    Parameters
    ----------
    y_pred : a numpy array
        representing the predictions returned by
        problem.workflow.test_submission; 1D (single target regression)
        or 2D (multi-target regression)
    y_true : a numpy array
        representing the ground truth returned by problem.get_train_data
        and problem.get_test_data; 1D (single target regression)
        or 2D (multi-target regression)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    if y_pred is not None:
        if fold_is is not None:
            y_pred = y_pred[fold_is]
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        if fold_is is not None:
            y_true = y_true[fold_is]
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        if self.n_columns == 0:
            shape = (n_samples)
        else:
            shape = (n_samples, self.n_columns)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()

def _generation_img_init(self, y_pred=None, y_true=None, n_samples=None,
                     fold_is=None):
    """Initialize a generation prediction type.
    The input is either y_pred, or y_true, or n_samples.
    Parameters
    ----------
    y_pred : a numpy array
        representing the predictions returned by
        problem.workflow.test_submission; 1D (single target regression)
        or 2D (multi-target regression)
    y_true : a numpy array
        representing the ground truth returned by problem.get_train_data
        and problem.get_test_data; 1D (single target regression)
        or 2D (multi-target regression)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    if y_pred is not None:
        if fold_is is not None:
            y_pred = y_pred[fold_is]
        self.y_pred = np.array(y_pred)
    elif y_true is not None:
        if fold_is is not None:
            y_true = y_true[fold_is]
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        shape = (n_samples, self.channels, self.height, self.width)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')
    self.check_y_pred_dimensions()


def make_generative(label_names=[]):
    """Creates a prediction type for generative challenges, based on regression.

    Args:
        label_names (list, optional): list of features. Defaults to [].

    Returns:
        Predictions: A prediction type for generative challenges.
    """
    Predictions = type(
        'Regression',
        (BasePrediction,),
        {'label_names': label_names,
         'n_columns': len(label_names),
         'n_columns_true': len(label_names),
         '__init__': _generation_init,
         })
    return Predictions

def make_generative_img(height, width, label_names=[]):
    Predictions = type(
        'Regression',
        (BaseImgGen,),
        {'label_names': label_names,
         'channels': 3,
         'height': height,
         'width': width,
         '__init__': _generation_img_init,
         })
    return Predictions
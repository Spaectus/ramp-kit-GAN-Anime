"""Generation predictions.
y_pred can be one or two-dimensional (for multi-target regression)
"""

# Author: MAI Huu Tan

import numpy as np
from collections.abc import Generator
from rampwf.prediction_types.base import BasePrediction

from workflows.image_generative import KnownLengthGenerator


class BaseImgGen(BasePrediction):
    def valid_indexes(self):
        raise NotImplementedError
        if len(self.y_pred.shape) <= 3:
            ValueError('y_pred.shape <= 3 is not implemented')
        elif len(self.y_pred.shape) == 4:
            # (n_samples, channel, height, width)
            return ~np.isnan(self.y_pred).any(axis=1)
        else:
            raise ValueError('y_pred.shape > 4 is not implemented')

    def check_y_pred_dimensions(self):
        assert self.channels is not None
        assert self.height is not None
        assert self.width is not None
        if not isinstance(self.y_pred, (Generator, KnownLengthGenerator, tuple, type(None))):
            raise ValueError(f"y_pred should be a generator or a tuple or NoneType, {type(self.y_pred)} found")

    def set_valid_in_train(self, predictions, test_is):
        """Set a cross-validation slice."""
        # This function is called for "Bagged scores"
        # No test data for our use case
        # print(f"in set_valid_in_train {test_is=}\n{type(predictions.y_pred)=}") # TODO
        #print(f"ICI {type(predictions)=}")
        self.y_pred = predictions.y_pred  # TODO
        # self.y_pred[test_is] = predictions.y_pred

    def set_slice(self, valid_indexes):
        """Collapsing y_pred to a cross-validation slice.
        So scores do not need to deal with masks.
        """
        #This function is called for "Bagged scores"
        #print(f"Set_slice {valid_indexes=}")
        pass  # TODO
        # self.y_pred = self.y_pred[valid_indexes]

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine predictions in predictions_list[index_list].
        The default implemented here is by taking the mean of their y_pred
        views. It can be overridden in derived classes.
        E.g. for regression it is the actual
        predictions, and for classification it is the probability array (which
        should be calibrated if we want the best performance). Called both for
        combining one submission on cv folds (a single model that is trained on
        different folds) and several models on a single fold (blending).
        Parameters
        ----------
        predictions_list : list of instances of Base
            Each element of the list is an instance of Base with the
            same length and type.
        index_list : None | list of integers
            The subset of predictions to be combined. If None, the full set is
            combined.
        Returns
        -------
        combined_predictions : instance of cls
            A predictions instance containing the combined predictions.
        """
        # print(f"combine {index_list=}")
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array(
            [predictions_list[i].y_pred for i in index_list])
        # print(f"{y_comb_list=}")
        # assert len(index_list) == 1
        return cls(y_pred=y_comb_list[0])  # TODO

        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(y_pred=y_comb)
        return combined_predictions


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
                         fold_is=None, channels=None, height=None, width=None):
    """Initialize a generation prediction type.
    The input is either y_pred, or y_true, or n_samples.
    Parameters
    ----------
    y_pred : a generator
    y_true :
        representing the ground truth returned by problem.get_train_data
        and problem.get_test_data; 1D (single target regression)
        or 2D (multi-target regression)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    assert y_pred is not None or y_true is not None or n_samples is not None
    assert isinstance(y_true, (tuple, type(None))), f"{type(y_true)=}"
    if isinstance(y_true, tuple):
        # hence y_true is tuple of path
        if fold_is is not None:
            # fold_is == list d'indices
            #print(f"{fold_is[:10]=}")
            pass
        # print(f"{len(y_true)}")
    if 0:
        print(f"{type(y_pred)=}")
        print(f"{type(y_true)=}")
    if 0:
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
    else:
        if y_pred is None:
            if y_true is None:
                self.y_pred = None  # TODO
            else:
                self.y_pred = y_true
        else:
            self.y_pred = y_pred
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


def make_generative_img(channels, height, width):
    Predictions = type(
        'Regression',
        (BaseImgGen,),
        {'channels': channels,
         'height': height,
         'width': width,
         '__init__': _generation_img_init,
         })
    return Predictions

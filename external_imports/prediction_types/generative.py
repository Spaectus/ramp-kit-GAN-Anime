import numpy as np
from collections.abc import Generator
from rampwf.prediction_types.base import BasePrediction

from workflows.image_generative import KnownLengthGenerator


class BaseImgGen(BasePrediction):
    def valid_indexes(self):
        """For our case (image generation), we do not need  to combine Predictions on different cross validation slices
        This method is never called
        """
        raise NotImplementedError

    def check_y_pred_dimensions(self):
        """In our case, y_pred is not a numpy matrix. So it doesn't make sense to check the dimensions of y_pred.
        We simply check its type.
        """
        assert self.channels is not None
        assert self.height is not None
        assert self.width is not None
        if not isinstance(self.y_pred, (Generator, KnownLengthGenerator, tuple, type(None))):
            raise ValueError(f"y_pred should be a generator or a tuple or NoneType, {type(self.y_pred)} found")

    def set_valid_in_train(self, predictions, test_is):
        """No need for this method in the case of image generation"""
        self.y_pred = predictions.y_pred

    def set_slice(self, valid_indexes):
        """No need of set_slice for the image genrative use case
        This function is called for "Bagged scores"
        """
        pass

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine predictions in predictions_list[index_list].
        The default implemented here is by taking the first prediction.

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
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array([predictions_list[i].y_pred for i in index_list])
        return cls(y_pred=y_comb_list[0])


def _generation_img_init(self, y_pred=None, y_true=None, n_samples=None,
                         fold_is=None, channels=None, height=None, width=None):
    """Initialize a generation image prediction type.
    The input is either y_pred, or y_true, or n_samples.
    Parameters
    ----------
    y_pred : a generator
    y_true : tuple of Path elements (Path instance from from the pathlib module)
    n_samples : int
        to initialize an empty container, for the combined predictions
    fold_is : a list of integers
        either the training indices, validation indices, or None when we
        use the (full) test data.
    """
    assert y_pred is not None or y_true is not None or n_samples is not None
    assert isinstance(y_true, (tuple, type(None))), f"{type(y_true)=}"

    if y_pred is None:
        if y_true is None:
            self.y_pred = None
        else:
            self.y_pred = y_true
    else:
        self.y_pred = y_pred
    self.check_y_pred_dimensions()

def make_generative_img(channels, height, width):
    """Creates a prediction type for generative challenges
        Parameters:
            channels: numbers of channels of images, 3 for colored images
            height: height of images in pixels
            width: width of images in peixels

        Returns:
            Predictions: A prediction type for generative challenges.
        """
    Predictions = type(
        'Regression',
        (BaseImgGen,),
        {'channels': channels,
         'height': height,
         'width': width,
         '__init__': _generation_img_init,
         })
    return Predictions

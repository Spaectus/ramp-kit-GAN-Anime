import collections
import os
from pathlib import Path
from typing import List
import itertools

from rampwf.utils.importing import import_module_from_source

import numpy as np


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ImageGenerative():
    """

    Worflow for image generation challenge

    the submissions must contain only one file "generative.py".
    This file must contain a "Generator" class which must contain the following methods:
    __init__(self, latent_space_dimension) : where latent_space_dimension is the dimension of the latent space
    fit(self, batchGeneratorBuilder) : where batchGeneratorBuilder is an instance to retrieve batchs of images
    generate(self, latent_space_noise) : where latent_space_noise is a noise matrix in the latent space

    Inspired from https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/workflows/image_classifier.py
    """

    def __init__(self, workflow_element_names=['generator'],
                 n_images_generated: int = 100,
                 latent_space_dimension: int = 1024,
                 y_pred_batch_size: int = 64,
                 chunk_size_feeder: int = 128,
                 seed: int = 23,
                 channels: int = 3, width: int = 64, height: int = 64,
                 n_jobs_batch_generator: int = -1, n_points_interpolate: int = 150):
        """

        :param n_points_interpolate: number of point on which we do the interpolation
        :param n_jobs_batch_generator: Number of workers converting dataset images on disk into numpy matrix.
        :param seed: integer value for reproductible results
        :param y_pred_batch_size: The generated images are not all in memory at the same time, they are put in batch.
        y_pred_batch_size is the size of these batch
        :param n_images_generated: number of images generated to evaluate a generator
        :param workflow_element_names: List of python file names contained in a submission
        :param latent_space_dimension: number of real numbers needed to describe a point in the latent space
        """

        self.channels = channels
        self.width = width
        self.height = height
        self.n_images_generated: int = n_images_generated
        self.latent_space_dimension: int = latent_space_dimension
        self.elements_names = workflow_element_names
        self.y_pred_batch_size = y_pred_batch_size
        self.chunk_size_feeder: int = chunk_size_feeder
        self.seed = seed
        self.n_jobs_batch_generator = n_jobs_batch_generator
        self.n_points_interpolate = n_points_interpolate

        self.rng = np.random.default_rng(seed=self.seed)

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        """Train a batch image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain generator.py.
        X_array : Tuple of dataset image paths
        y_array : Tuple of dataset image paths

        train_is : vector of int
           indices from X_array to train on
        """

        assert isinstance(X_array, tuple)  # tuple of paths
        assert isinstance(y_array, tuple)  # tuple of paths

        image_generator = import_module_from_source(
            os.path.join(module_path, self.elements_names[0] + ".py"),
            self.elements_names[0],
            sanitize=True
        )
        # We instantiate the generator of the submission
        generator = image_generator.Generator(latent_space_dimension=self.latent_space_dimension)

        # we retrieve selected images (of the current fold) with indices provided by train_is
        selected_images: List = [X_array[indice] for indice in train_is]

        # We convert selected_images (list of path) to BatchGeneratorBuilderNoValidNy
        folders = set(path_.parent.name for path_ in selected_images)  # we retrieve folders required by the data
        assert len(folders) == 1, f"They are not exactly one folder ({len(folders)}) {folders=}"
        folder = tuple(folders)[0]

        # print(f"Train on {folder=}")
        images_names = [str(path.absolute()) for path in selected_images]

        g = BatchGeneratorBuilderNoValidNy(images_names, f"data/{folder}", chunk_size=self.chunk_size_feeder,
                                           n_jobs=self.n_jobs_batch_generator)

        # We train the generator of the submission
        generator.fit(g)

        return generator

    def check_generator_result(self, res_numpy, batch_size):
        if not isinstance(res_numpy, np.ndarray):
            raise ValueError(f"Output of generate function must be a np.ndarray, {type(res_numpy)} found")
        if len(res_numpy.shape) != 4:
            raise ValueError(
                f"Output of the generate function must be a np.ndarray with exactly {4} dimensions, {len(res_numpy.shape)} found")
        if res_numpy.shape[0] != batch_size:
            raise ValueError(
                f"The first dimension of the np.array returned by the generate function must be {batch_size} in this case, found {res_numpy.shape[0]}")
        if res_numpy.shape[1:] != (self.channels, self.height, self.width):
            raise ValueError(
                f"Output of the generate function must have shape {(batch_size, self.channels, self.height, self.width)}, found {res_numpy.shape}")
        if np.isnan(res_numpy).any():
            raise ValueError(
                f"Output of the generate function must be a np.ndarray without nan, {np.isnan(res_numpy).sum()} nan found")
        if np.isinf(res_numpy).any():
            raise ValueError(
                f"Output of the generate function must be a np.ndarray without inf, {np.isinf(res_numpy).sum()} inf found")

    def test_submission(self, trained_model, X_array):
        """
        This function generates Gaussian noise and gives it to the generator of the submission to recover images.
        Test_submission doesn't return directly the images, but returns a python generator (on which we can make a for
        loop for example) giving access to image batches of size self.y_pred_batch_size


        :param trained_model: object that is returned by train_submission, in our case the generator
        :param X_array: Tuple of dataset image paths
        :return: generated images from generator
        """
        assert isinstance(X_array, tuple)

        generator = trained_model  # we retrieve the model trained by the train_submission

        def y_pred_generator():
            for i in range(0, self.n_images_generated, self.y_pred_batch_size):
                upper = min(i + self.y_pred_batch_size, self.n_images_generated)
                batch_size = upper - i
                # Gaussian noise is generated for the latent space for the batch and given to the generator
                res_numpy = generator.generate(self.rng.normal(size=(batch_size, self.latent_space_dimension)))
                self.check_generator_result(res_numpy, batch_size)
                yield res_numpy

            yield None  # indicate that we switch to interpolation
            z1, z2 = self.rng.normal(size=(2, self.latent_space_dimension))
            t_vec = np.linspace(0, 1, num=self.n_points_interpolate)

            # let's create a batch with the segment t * z1 + (1-t) * z2
            # segment_batch = np.array([t*z1 + (1-t) * z2 for t in t_vec])

            for i in range(0, len(t_vec), self.y_pred_batch_size):
                upper = min(i + self.y_pred_batch_size, self.n_points_interpolate)
                batch_size = upper - i
                # let's create a batch with the segment t * z1 + (1-t) * z2
                current_segment_batch = np.array([t * z1 + (1 - t) * z2 for t in t_vec[i:upper]])
                assert len(current_segment_batch) == batch_size, f"{current_segment_batch.shape=} and {batch_size=}"
                res_numpy = generator.generate(current_segment_batch)
                self.check_generator_result(res_numpy, batch_size)
                yield res_numpy

        # Instead of directly returning the y_pred_generator(). We pass through the KnownLengthGenerator class.
        # KnownLengthGenerator copies the generator behavior of y_pred_generator() perfectly. Because of internal
        # checking in ramp, len(y_pred) must return something, but this cannot be the case if y_pred is simply the
        # y_pred_generator(). With the line below, we set len(y_pred) = self.n_images_generated
        return KnownLengthGenerator(y_pred_generator(), self.n_images_generated)


# inspired from BatchGeneratorBuilder found here :
# https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/workflows/image_classifier.py
class BatchGeneratorBuilderNoValidNy():
    """A batch generator builder for generating images on the fly.
    This class is a way to build training generators that yield each time a mini-batches of images.

    An instance of this class is exposed to users through
    the `fit` function : model fitting is called by using
    "clf.fit(gen_builder)" where `gen_builder` is an instance
    of this class : `BatchGeneratorBuilder`.
    The fit function from `Generator` should then use the instance
    to build train generators, using the method `get_train_valid_generators`
    Parameters
    ==========
    X_array :
        vector of path (in string) to images
    folder : str
        folder where the images are
    chunk_size : int
        size of the chunk used to load data from disk into memory.
    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.
    """

    def __init__(self, X_array, folder, chunk_size, n_jobs):
        self.X_array = np.array(X_array)
        self.folder = folder
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.nb_examples = len(X_array)

    def get_train_generators(self, batch_size=256):
        """Build train and valid generators for keras.
        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.
        Parameters
        ==========
        batch_size : int
            size of mini-batches

        Returns
        =======
        gen_train, a generator function for training data
        """
        nb_train = self.nb_examples
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        return gen_train, nb_train

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)

        it = _chunk_iterator(
            X_array=self.X_array[indices], folder=self.folder,
            n_jobs=self.n_jobs)
        for X in it:
            # Yielding mini-batches
            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size]


def _chunk_iterator(X_array, folder, chunk_size=1024, n_jobs=8):
    """Generate chunks of images.
    Parameters
    ==========
    X_array : ArrayContainer of int
        image names to load
    chunk_size : int
        chunk size
    folder : str
        folder where the images are
    n_jobs : int
        number of jobs used to load images in parallel
    Yields
    ======

    The shape of each element of X is (color, height, width), where color is 1 or 3 or 4 and height/width
    do not vary among images.
    """
    from skimage.io import imread  # TODO
    from joblib import delayed
    from joblib import Parallel
    for i in range(0, len(X_array), chunk_size):
        X_chunk = X_array[i:i + chunk_size]
        filenames = [
            os.path.join(folder, '{}'.format(x))
            for x in X_chunk]
        X = Parallel(n_jobs=n_jobs, backend='threading')(delayed(imread)(
            filename) for filename in filenames)
        yield np.moveaxis(X, -1, 1)  # from (height, width, color) to (color, height, width)


class KnownLengthGenerator:
    def __init__(self, gen, length):
        self.gen = gen
        self.length = int(length)

    def __len__(self):
        return self.length

    def __iter__(self):
        yield from self.gen

    def __next__(self):
        print(f"Destruction of a generator !!")
        return next(self.gen)

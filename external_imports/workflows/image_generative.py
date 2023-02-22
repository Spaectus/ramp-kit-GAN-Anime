import collections
import os
from pathlib import Path
from typing import List

from rampwf.utils.importing import import_module_from_source

import numpy as np


class ImageGenerative():
    """


    Inspired from https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/workflows/image_classifier.py
    """

    def __init__(self, workflow_element_names=['generator'], n_images_generated: int = 100,
                 latent_space_dimension: int = 1024, y_pred_batch_size: int = 64, chunk_size_feeder: int = 128,
                 seed: int = 23):
        """

        :param n_images_generated: umber of images generated to evaluate a generator
        :param workflow_element_names:
        :param latent_space_dimension: number of real numbers needed to describe a point in the latent space
        """
        self.n_images_generated: int = n_images_generated
        self.latent_space_dimension: int = latent_space_dimension
        self.elements_names = workflow_element_names
        self.y_pred_batch_size = y_pred_batch_size
        self.chunk_size_feeder: int = chunk_size_feeder
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        """Train a batch image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain generator.py.
        X_array : ArrayContainer vector of int

        y_array : vector of int

        train_is : vector of int
           indices from X_array to train on
        """

        # Note : le type de X_df dépend du get_train_data in problem.py
        assert isinstance(X_array, tuple)  # X_array : tuple de path
        assert isinstance(y_array, tuple)  # tuple de path

        image_generator = import_module_from_source(
            os.path.join(module_path, self.elements_names[0] + ".py"),
            self.elements_names[0],
            sanitize=True
        )

        generator = image_generator.Generator(latent_space_dimension=self.latent_space_dimension)

        # we retrieve slected image with indices provided by train_is
        selected_images: List = [X_array[indice] for indice in train_is]

        # We convert selected_images (list of path) to BatchGeneratorBuilderNoValidNy
        folders = set(path_.parent.name for path_ in selected_images)  # we retrieve folders required by the data
        assert len(folders) == 1, f"They are not exactly one folder ({len(folders)}) {folders=}"
        folder = tuple(folders)[0]
        print(f"Train on {folder=}")
        images_names = [str(path.absolute()) for path in (Path("data") / folder).glob("*.jpg")]

        g = BatchGeneratorBuilderNoValidNy(images_names, f"data/{folder}", chunk_size=self.chunk_size_feeder, n_jobs=-1)

        generator.fit(g)
        return generator

    def test_submission(self, trained_model, X_array):
        """

        :param trained_model: object that is returned by train_submission, in our case the generator
        :param X_array: object that is returned by get_test_data in problem.py
        :return: generated images from generator
        """
        assert isinstance(X_array, tuple)

        # print(f"test_submission {X_array[:10]=}")


        generator = trained_model  # we retieve the model trained by the train_submission

        # Gaussian noise is generated for the latent space.
        latent_space_noise = self.rng.normal(size=(self.n_images_generated, self.latent_space_dimension))

        # return KnownLengthGenerator(generator.generate(latent_space_noise), self.n_images_generated)

        def gen():
            for i in range(0, self.n_images_generated, self.y_pred_batch_size):
                upper = min(i + self.y_pred_batch_size, self.n_images_generated)
                batch = latent_space_noise[i:upper]
                res_numpy = generator.generate(batch)
                if not isinstance(res_numpy, np.ndarray):
                    raise ValueError(f"Output of generate function must be a np.ndarray, {type(res_numpy)} found")
                if len(res_numpy.shape) != 4:
                    raise ValueError(
                        f"Output of the generate function must be a np.ndarray with exactly {4} dimensions, {len(res_numpy.shape)} found")
                if res_numpy.shape[0] != len(batch):
                    raise ValueError(
                        f"The first dimension of the np.array returned by the generate function must be {len(batch)} in this case, found {res_numpy.shape[0]}")
                if np.isnan(res_numpy).any():
                    raise ValueError(
                        f"Output of the generate function must be a np.ndarray without nan, {np.isnan(res_numpy).sum()} nan found")
                if np.isinf(res_numpy).any():
                    raise ValueError(
                        f"Output of the generate function must be a np.ndarray without inf, {np.isinf(res_numpy).sum()} inf found")

                yield res_numpy

        return KnownLengthGenerator(gen(), self.n_images_generated)


class BatchGeneratorBuilderNoValidNy():
    """A batch generator builder for generating images on the fly.
    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).
    An instance of this class is exposed to users `Classifier` through
    the `fit` function : model fitting is called by using
    "clf.fit(gen_builder)" where `gen_builder` is an instance
    of this class : `BatchGeneratorBuilder`.
    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`
    Parameters
    ==========
    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).
    folder : str
        folder where the images are
    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).
    n_classes : int
        Total number of classes. This is needed because the array
        of labels, which is a vector of ints, is transformed into
        a onehot representation.
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
        valid_ratio : float between 0 and 1
            ratio of validation data
        Returns
        =======
        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
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
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            it = _chunk_iterator(
                X_array=self.X_array[indices], folder=self.folder,
                n_jobs=self.n_jobs)
            for X in it:
                # 1) Preprocessing of X and y
                # X = Parallel(
                # n_jobs=self.n_jobs, backend='threading')(delayed(
                #     self.transform_img)(x) for x in X)
                # print("X")
                # print(f"{X[0].shape} {X[0].dtype=}")

                # X shape : 64 x 64 x 3
                X = [np.moveaxis(x, -1, 0) for x in X]  # TODO when X : np.array
                # W
                X = np.array(X)

                # 2) Yielding mini-batches
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size]


def _chunk_iterator(X_array, folder, chunk_size=1024, n_jobs=8):
    """Generate chunks of images, optionally with their labels.
    Parameters
    ==========
    X_array : ArrayContainer of int
        image ids to load
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).
    chunk_size : int
        chunk size
    folder : str
        folder where the images are
    n_jobs : int
        number of jobs used to load images in parallel
    Yields
    ======
    if y_array is provided (not None):
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is `chunk_size` at most (it can be smaller).
    if y_array is not provided (it is None)
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is `chunk_size` at most (it can be
        smaller).
        This is used for testing, where we don't have/need the labels.
    The shape of each element of X in both cases
    is (height, width, color), where color is 1 or 3 or 4 and height/width
    vary according to examples (hence the fact that X is a list instead of
    numpy array).
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
        yield X


class KnownLengthGenerator:
    def __init__(self, gen, length):
        self.gen = gen
        self.length = int(length)

    def __len__(self):
        return self.length

    def __iter__(self):
        from itertools import tee
        original, new = tee(self.gen, 2)
        yield from new
        #yield from self.gen

    def __next__(self):
        print(f"Destruction of a generator !!")
        return next(self.gen)

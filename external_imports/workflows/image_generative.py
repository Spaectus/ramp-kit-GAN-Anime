import os
from rampwf.utils.importing import import_module_from_source

import numpy as np

class ImageGenerative():
    """


    Inspired from https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/workflows/image_classifier.py
    """

    def __init__(self, workflow_element_names=['generator'], n_images_generated:int=100,latent_space_dimension:int=1024):
        """

        :param n_images_generated: umber of images generated to evaluate a generator
        :param workflow_element_names:
        :param latent_space_dimension: number of real numbers needed to describe a point in the latent space
        """
        self.n_images_generated:int = n_images_generated
        self.latent_space_dimension: int = latent_space_dimension
        self.elements_names = workflow_element_names

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        """Train a batch image classifier.
        module_path : str
            module where the submission is. the folder of the module
            have to contain generator.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        # Note : le type de X_df dépend du get_train_data in problem.py

        image_generator = import_module_from_source(
            os.path.join(module_path, self.elements_names[0] + ".py"),
            self.elements_names[0],
            sanitize=True
        )

        generator = image_generator.Generator(latent_space_dimension=self.latent_space_dimension)
        generator.fit(X_df)
        return generator

    def test_submission(self, trained_model, X_array):
        """

        :param trained_model: object that is returned by train_submission, in our case the generator
        :param X_array: object that is returned by get_test_data in problem.py
        :return: generated images from generator
        """
        #print(f"{X_array.shape=}")
        generator = trained_model # we retieve the model trained by the train_submission
        # Gaussian noise is generated for the latent space.
        latent_space_noise = np.random.normal(size=(self.n_images_generated, self.latent_space_dimension))

        batch_size = 64

        for i in range(0, self.n_images_generated, batch_size):
            upper = min(i + batch_size, self.n_images_generated)
            batch = latent_space_noise[i:upper]
            res_numpy = generator.generate(batch)

            yield res_numpy



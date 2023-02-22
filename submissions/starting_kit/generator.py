import torchvision
import numpy as np

class Generator():
    """
    This a dumny generator
    """

    def __init__(self, latent_space_dimension):
        self.memory = None
        self.latent_space_dimension: int = latent_space_dimension

    # def fit(self, image_folder: torchvision.datasets.ImageFolder):
    def fit(self, batchGeneratorBuilderNoValidNy):
        generator_of_images, total_nb_images = batchGeneratorBuilderNoValidNy.get_train_generators(batch_size=100)
        self.memory = next(generator_of_images)  # grab the first mini-batch of images.
        # self.memory = images_matrix[: 100]  # we memorise 100 images of the train dataset
        assert isinstance(self.memory, np.ndarray)

    def generate(self, latent_space_noise):
        """
        Generates nb_image colored (3 channels : RGB) images of size 64 x 64
        from a matrix in the latent space.

        :param latent_space_noise: matrix of dimension (nb_image, 1024) issued from a normal (Gaussian) distribution
        :return: matrix of size (nb_image, 3, 64, 64)
        """
        nb_image = latent_space_noise.shape[0]
        assert len(
            self.memory) >= nb_image, f"We do not saved enough images ! We saved {len(self.memory)} but we need {nb_image}"

        assert isinstance(self.memory, np.ndarray)

        return self.memory[:nb_image]

        #for i in range(nb_image):
        #    assert self.memory[i].shape == (3, 64, 64), f"{self.memory[i].shape=}"
        #    yield self.memory[i]

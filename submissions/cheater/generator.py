import torchvision
import numpy as np

class Generator():
    """
    This a cheater generator that returns samples from the training dataset.
    """

    def __init__(self, latent_space_dimension):
        self.memory = None
        self.latent_space_dimension: int = latent_space_dimension

    # def fit(self, image_folder: torchvision.datasets.ImageFolder):
    def fit(self, batchGeneratorBuilderNoValidNy):

        generator_of_images, total_nb_images = batchGeneratorBuilderNoValidNy.get_train_generators(batch_size=100)
        self.memory = next(generator_of_images)  # grab the first mini-batch of images. We can't memorize too many images.
        assert isinstance(self.memory, np.ndarray)

    def generate(self, latent_space_noise):
        """Generates a minibatch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.

        Args:
            latent_space_noise (ndarray, shape (nb_image, 1024)): a mini-batch of noise of dimension  issued from a normal (Gaussian) distribution

        Returns:
            ndarray, shape (nb_image, 3, 64, 64): A mini-batch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.
        """
        nb_image = latent_space_noise.shape[0]

        # assert len(
        #     self.memory) >= nb_image, f"We do not saved enough images ! We saved {len(self.memory)} but we need {nb_image}"

        assert isinstance(self.memory, np.ndarray)
        rng = np.random.RandomState(seed=0)

        return rng.choice(self.memory, size=nb_image, replace=True)


import torchvision
import numpy as np

class Generator():
    """
    This a dummy generator that generates noise of size (3, 64, 64).
    """

    def __init__(self, latent_space_dimension):
        pass

    def fit(self, batchGeneratorBuilderNoValidNy):
        pass

    def generate(self, latent_space_noise):
        """Generates a minibatch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.

        Args:
            latent_space_noise (ndarray, shape (nb_image, 1024)): a mini-batch of noise of dimension  issued from a normal (Gaussian) distribution

        Returns:
            ndarray, shape (nb_image, 3, 64, 64): A mini-batch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.
        """
        nb_image = latent_space_noise.shape[0]
        # Generate random samples
        rng = np.random.RandomState(seed=0)
        res = np.abs(rng.randn(nb_image, 3, 64, 64))
        res = res/res.max()

        return res

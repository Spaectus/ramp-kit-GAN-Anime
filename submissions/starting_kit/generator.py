import numpy as np


class Generator():
    """
    This a dummy generator that generates noise of size (3, 64, 64).
    """

    def __init__(self, latent_space_dimension):
        """Initializes a Generator object that is used for `ramp` training and evaluation.
        
        This object is used to wrap your generator model and anything else required to train it and
        to generate samples.

        In this submission, only a rng is required.

        Args:
            latent_space_dimension (int): Dimension of the latent space, where inputs of the generator are sampled.
        """
        self.rng = np.random.RandomState(seed=0)

    def fit(self, batchGeneratorBuilder):
        """Trains the generator with a batch generator builder, which can return a Python Generator with its method `get_train_generators`.

        In this submission, this method does nothing.

        Args:
            batchGeneratorBuilder (_type_): _description_
        """
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

        res = np.abs(self.rng.randn(nb_image, 3, 64, 64))
        res = res / res.max()

        return res

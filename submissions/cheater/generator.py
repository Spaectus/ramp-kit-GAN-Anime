import torchvision
import numpy as np

class Generator():
    """
    This a cheater generator that returns samples from the training dataset.
    """

    def __init__(self, latent_space_dimension):
        """Initializes a Generator object that is used for `ramp` training and evaluation.
        
        This object is used to wrap your generator model and anything else required to train it and
        to generate samples.

        In this submission, only a memory is required to save some training samples, as well as a memory limit.
        We will also use a rng to randomly pick samples that we return.

        Args:
            latent_space_dimension (int): Dimension of the latent space, where inputs of the generator are sampled.
        """
        self.memory = None
        self.max_samples = 500
        self.rng = np.random.RandomState(seed=0)

    # def fit(self, image_folder: torchvision.datasets.ImageFolder):
    def fit(self, batchGeneratorBuilderNoValidNy):
        """Trains the generator with a batch generator builder, which can return a Python Generator with its method `get_train_generators`.

        In this submission, this method saves the first batch in memory.

        Args:
            batchGeneratorBuilderNoValidNy (_type_): _description_
        """

        generator_of_images, total_nb_images = batchGeneratorBuilderNoValidNy.get_train_generators(batch_size=100)
        memory = []
        size = 0
        while size < self.max_samples:
            batch = next(generator_of_images)
            if batch is None:
                break
            size_batch = batch.shape[0]
            taken = min(self.max_samples - size, size_batch)
            memory.append(batch[:taken].copy())
            size += size_batch
        self.memory = np.concatenate(memory, axis=0)  # grab the first mini-batch of images. We can't memorize too many images.
        assert isinstance(self.memory, np.ndarray)

    def generate(self, latent_space_noise):
        """Generates a minibatch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.

        In this submission, the method returns random samples of the memory (which contains training samples) with replacement.

        Args:
            latent_space_noise (ndarray, shape (nb_image, 1024)): a mini-batch of noise of dimension  issued from a normal (Gaussian) distribution

        Returns:
            ndarray, shape (nb_image, 3, 64, 64): A mini-batch of `nb_image` colored (3 channels : RGB) images of size 64 x 64 from a matrix in the latent space.
        """
        nb_image = latent_space_noise.shape[0]

        # assert len(
        #     self.memory) >= nb_image, f"We do not saved enough images ! We saved {len(self.memory)} but we need {nb_image}"

        assert isinstance(self.memory, np.ndarray)
        memory_size = self.memory.shape[0]
        idx = np.arange(memory_size)

        chosen_idx = self.rng.choice(idx, size=nb_image, replace=True)
        return self.memory[chosen_idx, :, :, :]


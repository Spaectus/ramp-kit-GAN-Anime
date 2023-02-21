import torchvision


class Generator():
    """
    This a dumny generator
    """

    def __init__(self, latent_space_dimension):
        self.memory = None
        self.latent_space_dimension: int = latent_space_dimension

    # def fit(self, image_folder: torchvision.datasets.ImageFolder):
    def fit(self, images_matrix):
        self.memory = images_matrix[: 100]  # we memorise 100 images of the train dataset

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
        # We don't care of latent_space_noise, we just return the memorised images
        y_pred = self.memory[: nb_image]
        return y_pred

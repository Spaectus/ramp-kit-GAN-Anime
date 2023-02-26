from collections import Counter
import itertools

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from rampwf.score_types.base import BaseScoreType

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from PIL import Image

import gc

import warnings


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def disable_torchmetrics_warnings():
    """This function disables the warnings due to initializing KernelInceptionDistance objects from torchmetrics.image.kid.
    """
    warnings.resetwarnings()
    warnings.filterwarnings('ignore')  # Ignore everything
    # ignore everything does not work: ignore specific messages, using regex
    warnings.filterwarnings(
        'ignore', '.*UserWarning: Metric `Kernel Inception Distance`*')


disable_torchmetrics_warnings()

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
    ToTensor(),
])


class ImageSet(Dataset):
    """This class inherits from the Dataset class of PyTorch and is used to load the images locally with the paths of the images.

    The images are already transformed beforehand, so all there is to do in order to feed them to the metrics is to send them in
    minibatches using a DataLoader object.
    """

    def __init__(self, paths, transform, preload=False):
        """Initializes the dataset from a tuple of paths.

        Args:
            paths (tuple of `str` objects): tuple of strings containing the paths of the images used in the Dataset.
            transform (Compose): A composition of transforms to be applied on the images.
            preload (bool, optional): A boolean to indicate whether the images are preloaded as PyTorch Tensor objects. Defaults to False.
        """
        self.paths = paths
        self.transform = transform
        self.preload = preload
        if self.preload:
            self.files = [
                self.transform(
                    Image.open(path)
                ) for path in self.paths]

    def __getitem__(self, index):
        """Gets an item from the dataset.

        Args:
            index (int): The index of the image in the dataset.

        Returns:
            Tensor: the `index`-th image in the dataset.
        """
        if self.preload:
            return self.files[index]
        else:
            return self.transform(
                Image.open(self.paths[index])
            )

    def __len__(self):
        """Returns the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.paths)


class Master():
    """A class that centralizes the computations of the metrics for `ramp-test`.

    Since images are generated by batch from the competitor's generator, a Python generator is used to retrieve the
    batches of images, which are then fed to the `Torchvision` objects to compute the metrics.

    In order to ensure that we only do one pass of each generated dataset, multiple metrics are computed at the same time, from
    the local fold metric to the bagged metrics that account for all the train sets.

    This class centralizes the computations of the metrics, which are then retrieved by the BaseScoreType objects of
    `ramp-workflow`.
    """

    def __init__(self, n_fold=3):
        """Initializes a Master object to centralize the computation of metrics.

        The object locally stores a FrechetInceptionDistance object, a KernelInceptionDistance and a InceptionScore object to
        compute the bagged metrics after making a full pass on each generated set.

        In order to keep track of the current fold, `memory` keeps track of the current fold for the metrics to compute.

        Args:
            n_fold (int, optional): The number of folds to perform for evaluation. Defaults to 3.
        """
        self.batch_size = 32
        self.score = {}
        # [None, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        self.pattern = [
            None] + [i for i in range(n_fold) for z in range(3)] + 2 * n_fold * [n_fold]  # 6
        self.memory_call = Counter()
        self.memory = Counter()
        self.n_fold = n_fold

        # Permanent metrics to compute bagged scores
        self.fid = FrechetInceptionDistance(
            reset_real_features=True, normalize=True).to(device)
        self.kid = KernelInceptionDistance(
            reset_real_features=True, normalize=True).to(device)
        self.is_ = InceptionScore(normalize=True).to(device)

    def eval(self, y_true, y_pred, metric):
        """Evaluates scores for a given metric on a certain fold.

        If no score has been computed for the current fold before, the method will compute all the scores for the current fold
        and store them in `self.scores`.

        This allows the metrics to be computed with a single pass on each generated set.

        Args:
            y_true (tuple of `str` objects): The paths of the images that are used as real images.
            y_pred (Generator): A Python generator object that yields mini-batches of generated samples as numpy arrays.
            metric (str): {"FID", "KID_mean", "KID_std", "IS_mean", "IS_std"} The name of the metric to compute.

        Returns:
            float: The value of the metric to compute for the current fold.
        """

        assert metric in ("FID", "KID_mean", "KID_std",
                          "IS_mean", "IS_std", "L1_norm_interpolation")
        self.memory_call[metric] += 1
        # retrieve position in k_fold
        current_fold: int = self.pattern[self.memory_call[metric]]
        self.current_fold = current_fold

        context = (metric, current_fold)
        self.memory[context] += 1  # we count the number of call of each metric

        if context in self.score:
            # We have already compute this metric for this fold
            return self.score[context]

        if current_fold == self.n_fold:
            # Compute the permanent metrics that received a full pass of each dataset and generated dataset.

            fid_score = self.fid.compute().item()
            self.score[("FID", current_fold)] = fid_score

            kid_mean, kid_std = self.kid.compute()
            # We rescale the KID scores because otherwise they are too small and too close to 0.
            self.score[("KID_mean", current_fold)] = kid_mean.item()
            self.score[("KID_std", current_fold)] = kid_std.item()

            is_mean, is_std = self.is_.compute()
            self.score[("IS_mean", current_fold)] = is_mean.item()
            self.score[("IS_std", current_fold)] = is_std.item()

            # For the interpolation score, we just pick the mean of the interpolation scores.
            self.score[("L1_norm_interpolation", current_fold)] = np.mean(
                [self.score[("L1_norm_interpolation", k)] for k in range(self.n_fold)])

            return self.score[context]

        fid = FrechetInceptionDistance(
            reset_real_features=True, normalize=True).to(device)
        kid = KernelInceptionDistance(
            reset_real_features=True, normalize=True).to(device)
        is_ = InceptionScore(normalize=True).to(device)

        # Handling generated data
        itertaor_enumerated_batch = enumerate(y_pred)
        for i, batch in itertaor_enumerated_batch:
            if batch is None:
                # A None batch indicates that the next batch if for the interpolate test
                # print(f"batch is now None, break")
                break

            batch_ = torch.Tensor(batch).to(device)

            if i == 0:
                displayed = vutils.make_grid(
                    batch_, padding=2, normalize=True).cpu()
                if displayed is not None:
                    plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.title("Generated Images")
                    plt.imshow(np.transpose(displayed, (1, 2, 0)))
                    print(
                        "The first batch of images is displayed on a different window. Please close it to continue evaluation.")
                    plt.show()

            fid.update(batch_, real=False)
            kid.update(batch_, real=False)
            is_.update(batch_)
            self.fid.update(batch_, real=False)
            self.kid.update(batch_, real=False)
            self.is_.update(batch_)

        # Now wa have to calculate the interpolation score
        self.score[("L1_norm_interpolation", current_fold)] = self.do_interpolation(
            itertaor_enumerated_batch, tol=0)

        # Handling true data
        folders = set(path_.parent.name for path_ in y_true)
        assert len(folders) == 3
        y_true_ = tuple(
            path for path in y_true if path.parent.name == f"train_{current_fold + 1}")

        dataset = ImageSet(
            paths=y_true_,
            transform=transform,
            preload=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        for batch in loader:
            batch_ = batch.to(device)
            fid.update(batch_, real=True)
            kid.update(batch_, real=True)
            self.fid.update(batch_, real=True)
            self.kid.update(batch_, real=True)

        fid_score = fid.compute().item()
        self.score[("FID", current_fold)] = fid_score

        kid_mean, kid_std = kid.compute()
        # We rescale the KID scores because otherwise they are too small and too close to 0.
        self.score[("KID_mean", current_fold)] = kid_mean.item()
        self.score[("KID_std", current_fold)] = kid_std.item()

        is_mean, is_std = is_.compute()
        self.score[("IS_mean", current_fold)] = is_mean.item()
        self.score[("IS_std", current_fold)] = is_std.item()

        # Delete models to make some space on the GPU.
        del fid, kid, is_
        torch.cuda.empty_cache()
        gc.collect()

        return self.score[context]

    def do_interpolation(self, remaining_y_pred, tol=.5):

        scores = []

        display_iterpolation = True  # display with matplotlib the interpolation images

        images = []

        previous_img = None

        for i, interpolate_batch in remaining_y_pred:
            if previous_img is not None:
                # we add
                iterator = pairwise(np.concatenate(
                    (np.expand_dims(previous_img, axis=0), interpolate_batch), axis=0))
            else:
                # first iteration in this for loop
                iterator = pairwise(interpolate_batch)
            for j, (img_1, img_2) in enumerate(iterator):
<<<<<<< HEAD
=======
                # scores.append(adjusted_mutual_info_score(img_1.ravel(), img_2.ravel()))
>>>>>>> d5837a737b799764fbdb1fab620ff8be547121df
                scores.append(np.abs(img_1 - img_2).mean())
                if display_iterpolation:
                    images.append(img_1)
                previous_img = img_2
        scores = np.array(scores)
        score = scores.max()

        if display_iterpolation:
            displayed = vutils.make_grid(
                torch.Tensor(np.stack(images, axis=0)).to(device), padding=2, normalize=True).cpu()
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Interpolation test (score = {score})")
            plt.imshow(np.transpose(displayed, (1, 2, 0)))
            print(
                "The interpolation test is displayed on a different window. Please close it to continue evaluation.")
            plt.show()

        return np.maximum(score - tol, 0)



MASTER = Master()


# Fréchet Inception Distance (FID)


class FID(BaseScoreType):
    precision = 5

    def __init__(self, name="FID"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="FID")


# Kernel Inception Distance (KID)

class KIDMean(BaseScoreType):
    precision = 5

    def __init__(self, name="KID_mean"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="KID_mean")


class KIDStd(BaseScoreType):
    precision = 5

    def __init__(self, name="KID_std"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="KID_std")


# Inception Score (IS)

class ISMean(BaseScoreType):
    precision = 5

    def __init__(self, name="IS_mean"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="IS_mean")


class ISStd(BaseScoreType):
    precision = 5

    def __init__(self, name="IS_std"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="IS_std")


class L1_norm(BaseScoreType):
    """This score is an interpolation score that is meant to detect cheating.

    By picking z1, z2 in the latent space, we can compute a pairwise distance score
    between the generated images G(t z1 + (1-t)z2) for t in [0, 1].

    Since the neural network should be continuous, this is a cheat detection that should
    rule out submissions that only sample from the training dataset to "generate".
    """
    precision = 7

    def __init__(self, name="L1_norm_interpolation"):
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        return MASTER.eval(y_true, y_pred, metric="L1_norm_interpolation")


class Mixed(BaseScoreType):
    """This score accounts for all the previous scores, and penalizes with the cheating detection.

    Formula:
                    alpha * IS_mean + beta * FID + gamma * KID_mean
    mixed =     ------------------------------------------------------
                            1 + delta * L1_norm_interpolation
    """
    precision = 3

    def __init__(self, name="mixed", alpha=1., beta=-1., gamma=-1., delta=1.):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        # Mixed is the last score to be computed, so we can be sure that
        # the scores below have already been computed.
        is_mean = MASTER.score[("IS_mean", MASTER.current_fold)]
        fid = MASTER.score[("FID", MASTER.current_fold)]
        kid_mean = MASTER.score[("KID_mean", MASTER.current_fold)]
        interpolation_score = MASTER.score[(
            "L1_norm_interpolation", MASTER.current_fold)]

        return np.maximum(0, self.alpha * is_mean + self.beta * fid + self.gamma * kid_mean)/(1 + self.delta * interpolation_score)

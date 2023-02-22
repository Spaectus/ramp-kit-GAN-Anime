from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torch import from_numpy
from rampwf.score_types.base import BaseScoreType

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = Compose([
    ToTensor(),
])



class ImageSet(Dataset):
    def __init__(self, paths, transform, preload=False):
        self.paths = paths
        self.transform = transform
        self.preload = preload
        if self.preload:
            self.files = [
                self.transform(
                    Image.open(path)
                ) for path in self.paths]

    def __getitem__(self, index):
        if self.preload:
            return self.files[index]
        else:
            return self.transform(
                Image.open(self.paths[index])
            )

    def __len__(self):
        return len(self.paths)

MEM= 1

class Troll(BaseScoreType):
    precision = 1

    def __init__(self, name="troll"):
        self.name = name
    def check_y_pred_dimensions(self, y_true, y_pred):
        pass
    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        assert len(y_true)
        if MEM and 0:
            print(f"{y_true[:4]=}")
            folders = set(path_.parent.name for path_ in y_true)
            print(f"{folders=}")
        return 1.

# Fréchet Inception Distance (FID)
class FID(BaseScoreType):
    precision = 2

    def __init__(self, name='fid_score'):
        # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        self.name = name
        self.batch_size = 32
        self.score = None

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        assert isinstance(y_true, tuple)
        # y_pred is generator with len
        fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)

        i = -1
        # Handling generated data
        for i, batch in enumerate(y_pred):
            batch_ = torch.Tensor(batch / 255).to(device)
            # print(batch_.size())
            fid.update(batch_, real=False)
        # If the generator is empty, it means that we already went through
        # this dataset.
        if i == -1 and not self.score is None:
            print("Generator is empty. We reuse the same previous score.")
            return self.score

        # Handling true data
        dataset = ImageSet(
            paths=y_true,
            transform=transform,
            preload=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        for batch in loader:
            batch_ = batch.to(device)
            # print(batch_)
            fid.update(batch_, real=True)
        # Compute score
        self.score = fid.compute().item()

        # Destroy the former instance of FID
        # self.fid = FrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)
        return self.score


# Kernel Inception Distance (KID)

# TODO: see if we can combine both into one type of score because computing the core twice
# is expensive.

class KID(object):
    def __init__(self, name='kid_mean'):
        self.kid = KernelInceptionDistance(reset_real_features=True).to(device)
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        for batch in y_true:
            batch_ = torch.Tensor(batch_).to(device)
            self.kid.update(batch_, real=True)
        for batch in y_pred:
            batch_ = torch.Tensor(batch).to(device)
            self.kid.update(batch_, real=False)
        score = self.kid.compute()
        # score = (mean, std)
        return score[0]


class KID(object):
    def __init__(self, name='kid_mean'):
        self.kid = KernelInceptionDistance(reset_real_features=True).to(device)
        self.name = name

    def check_y_pred_dimensions(self, y_true, y_pred):
        pass

    def __call__(self, y_true, y_pred):
        # y_true = X ; y_pred = X_gen = G(z)
        for batch in y_true:
            batch_ = torch.Tensor(batch_).to(device)
            self.kid.update(batch_, real=True)
        for batch in y_pred:
            batch_ = torch.Tensor(batch).to(device)
            self.kid.update(batch_, real=False)
        score = self.kid.compute()
        # score = (mean, std)
        return score[0]

# Inception Score

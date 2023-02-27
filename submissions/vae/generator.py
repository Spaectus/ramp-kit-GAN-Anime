import torchvision
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import torch.nn.functional as F

import urllib.request
from pathlib import Path
import zipfile

seed = 0
torch.manual_seed(seed)

def download_pretrained_weights():
    """This function downloads the weights of our pre-trained DCGAN over 50 epochs.

    This submission fine-tunes our model that was already pre-trained on many images.
    """
    DIR = Path(__file__).parent / "models"
    DIR.mkdir(exist_ok=True)

    DIR_VAE = DIR / "vae_4980.pth"
    # DIR_DEC = DIR / "decoder_19900.pth"

    if DIR_VAE.exists() :# and DIR_DEC.exists():
        return
    print("Did not find pretrained weights in the directory. Starting download.")
    tmp_filename = "weights_50_epochs.zip"

    url = "https://drive.rezel.net/s/kPnS6qxNmratt9Z/download/vae_4980.zip"

    urllib.request.urlretrieve(url, tmp_filename)
    with zipfile.ZipFile(tmp_filename, 'r') as zip_ref:
        zip_ref.extractall(DIR)

    Path(tmp_filename).unlink()
    print("Finished downloading.")


#@title Model and training parameters

training_path = "./data/train_1/"

batch_size = 128
image_size = 64
nb_channels = 3

latent_dim = 256

epochs = 20
learning_rate = 1e-4
KL_weight = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


#@title Dataset class and transforms
training_transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class ImageSet(Dataset):
    def __init__(self, path, transform, preload=False):
        self.path = path
        self.files = os.listdir(self.path)
        self.transform = transform
        self.preload = preload
        if self.preload:
          self.files = [
              self.transform(
                  Image.open(os.path.join(self.path, file))
              ) for file in self.files]

    def __getitem__(self, index):
        if self.preload:
          return self.files[index]
        else:
          return self.transform(
              Image.open(os.path.join(self.path, self.files[index]))
          )
    
    def __len__(self):
        return len(self.files)
    
#@title Datasets making
training_set = ImageSet(
    path=training_path,
    transform=training_transform,
    preload=True,
)


training_loader = DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True
)

class VAE(nn.Module):


    def __init__(self, nb_channels, n_features, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = nb_channels
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= nb_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return  x_hat, mu, log_var
    



class Generator():
    def __init__(self):
        self.latent_space_dimension: int = 100

        self.batch_size = 128
        self.image_size = 64
        self.channels = 3

        self.g_features = 64
        self.d_features = 64
        self.n_features = 64

        self.epochs = 1
        self.lr = 2e-4
        self.beta1 = 0.5

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models
        self.VAE = VAE(self.channels, self.n_features, self.latent_space_dimension)
        # Optimizers
        # self.criterion = vae_loss()
        self.optimizer = optim.Adam(self.VAE.parameters(), lr=self.learning_rate)
        download_pretrained_weights()

    def fit(self,batchGeneratorBuilderNoValidNy):


        PATH = Path(__file__).parent / "models"
        self.VAE.load_state_dict(
            torch.load(PATH / "vae_4980.pth"))

        self.VAE.to(self.device)

        self.VAE.train()


        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, x in enumerate(training_loader):
                # Move the input data to the same device as the model
                x = x.to(device)

                self.optimizer.zero_grad()
                x_hat, mu, log_var = self.VAE(x)
                loss = vae_loss(x, x_hat, mu, log_var, KL_weight)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, loss={total_loss / len(training_loader.dataset):.4f}")

    def generate(self):
        self.generator.eval()
        with torch.no_grad():
            random_noise = torch.randn(64, latent_dim, device=device)
            batch = self.VAE.decode(random_noise)

        return batch.numpy(force=True)




 
# Create an instance of the VAE model
# model = VAE(nb_channels, image_size, latent_dim).to(device)
 
def vae_loss(x, x_hat, mu, log_var, beta):
    # Reconstruction loss
    #pdb.set_trace()
    #recon_loss = nn.BCELoss(reduction='mean')(x_hat, x)
    recon_loss = F.mse_loss(x, x_hat)

    #recon_loss = (x * torch.log(x_hat) + (1 - x)* torch.log(1 - x_hat)).sum((1, 2, 3)).mean()
    #pdb.set_trace()
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #pdb.set_trace()
    # Total loss
    loss = recon_loss + beta * kld_loss
    
    return loss




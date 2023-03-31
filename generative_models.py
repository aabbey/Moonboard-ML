import sys

import torch
from torch import nn
import predictions_models


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, dim_0, dim_1):
        super(Trim, self).__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1

    def forward(self, x):
        return x[:, :, self.dim_0:, self.dim_1:]



class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,
                                   output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride
                                   ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels,
                                   output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, kernel_size=2, stride=1),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=kernel_size,
                          stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=kernel_size,
                          stride=stride)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class VAELoss(nn.Module):
    def __init__(self, recon_weight):
        super(VAELoss, self).__init__()
        self.recon_weight = recon_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, fake, real, z_log_var, z_mean):
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=1)
        kl_div = kl_div.mean()

        recon_loss = self.mse_loss(fake, real)
        loss = self.recon_weight * recon_loss + kl_div
        return loss


class VAEConv(nn.Module):
    def __init__(self, hidden, z_dim):
        super(VAEConv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.z_mean = nn.Linear(hidden * 9 * 6, z_dim)
        self.z_log_var = nn.Linear(hidden * 9 * 6, z_dim)

        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim+14, hidden * 9 * 6),
            Reshape(shape=(-1, hidden, 9, 6)),
            nn.ConvTranspose2d(hidden,
                               hidden,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden,
                               hidden,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden,
                               hidden,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden,
                               1,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            Trim(0, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mean, log_var, var_weight=1.):
        N = torch.randn(log_var.size(0), log_var.size(1), device=log_var.get_device())
        z = mean + var_weight * torch.exp(log_var / 2.) * N
        return z

    def encoding_fn(self, x, var_weight=1.):
        mean, log_var = self.z_mean(self.encoder(x)), self.z_log_var(self.encoder(x))
        return self.reparameterize(mean, log_var, var_weight)

    def forward(self, x, grade, var_weight=1.):
        encoder_last_layer = self.encoder(x)
        mean, log_var = self.z_mean(encoder_last_layer), self.z_log_var(encoder_last_layer)
        encoded = self.reparameterize(mean, log_var, var_weight)
        decoded = self.decoder(torch.cat((encoded, grade), dim=1))
        return encoded, mean, log_var, decoded


class AutoencoderCNN1(nn.Module):
    def __init__(self, hidden, latent_dim):
        super(AutoencoderCNN1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(hidden*9*6, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden*9*6),
            Reshape(shape=(-1, hidden, 9, 6)),
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden, 1, kernel_size=3, stride=1, padding=1),
            Trim(dim_0=0, dim_1=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded

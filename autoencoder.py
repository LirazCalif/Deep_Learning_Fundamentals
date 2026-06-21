import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # Layer 1: 5x5 conv, 64 filters, downsampling, BNorm, ReLU
        # Input: (in_channels) x 64 x 64 -> Output: 64 x 32 x 32
        modules.append(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU())

        # Layer 2: 5x5 conv, 128 filters, downsampling, BNorm, ReLU
        # Input: 64 x 32 x 32 -> Output: 128 x 16 x 16
        modules.append(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())

        # Layer 3: 5x5 conv, 256 filters, downsampling, BNorm, ReLU
        # Input: 128 x 16 x 16 -> Output: 256 x 8 x 8
        modules.append(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())

        # Layer 4: 1x1 conv
        # Input: Output: 256 x 8 x 8 -> Output: (out_channels) x 1 x 1
        # modules.append(nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0))
        modules.append(nn.Conv2d(256, out_channels, kernel_size=8, stride=1, padding=0))
        modules.append(nn.ReLU())
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)

class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # reverse Layer 4
        # Input: (in_channels) x 1 x 1 -> Output: 256 x 8 x 8
        modules.append(nn.ConvTranspose2d(in_channels, 256, kernel_size=8, stride=1, padding=0))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        # reverse Layer 3
        # Input: 256 x 8 x 8 -> Output: 128 x 16 x 16
        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.ReLU())
        # reverse Layer 2
        # Input: 128 x 16 x 16 -> Output: 64 x 32 x 32
        modules.append(nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU())
        # reverse Layer 1
        # Input: 64 x 32 x 32 -> Output: (out_channels) x 1 x 1
        modules.append(nn.ConvTranspose2d(64, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))

class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder that extracts features from an input.
        :param features_decoder: Instance of a decoder that reconstructs an input from its features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)
        self.mu_layer = nn.Linear(n_features, z_dim)
        self.log_sigma2_layer = nn.Linear(n_features, z_dim)

        self.z2h = nn.Linear(z_dim, n_features)
    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        h = self.features_encoder(x)
        h_flatten = h.reshape(h.shape[0], -1)

        mu = self.mu_layer(h_flatten)
        log_sigma2 = self.log_sigma2_layer(h_flatten)

        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_sigma2)
        return z, mu, log_sigma2

    def decode(self, z):
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        h_flatten = self.z2h(z)
        h = h_flatten.reshape(h_flatten.shape[0], *self.features_shape)
        x_rec = self.features_decoder(h)
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            z = torch.randn(n, self.z_dim, device=device)
            samples = list(self.decode(z))
        # Detach and move to CPU for display purposes.
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2

def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # parameters
    N, C, H, W = x.shape
    d_x = C * H * W
    d_z = z_mu.shape[1]
    # data loss term
    diff_x = ((x-xr)**2).reshape(N,-1)
    data_loss = (1/(x_sigma2 * d_x)) * torch.sum(diff_x, 1)
    data_loss = data_loss.mean()
    # KL divergence loss
    z_sigma2 = torch.exp(z_log_sigma2)
    trace_sigma = z_sigma2.sum(dim=1)
    norm_mu = (z_mu**2).sum(dim=1)
    log_det_sigma = z_log_sigma2.sum(dim=1)
    kldiv_loss = trace_sigma + norm_mu - d_z - log_det_sigma
    kldiv_loss = kldiv_loss.mean()
    # VAE loss
    loss = data_loss + kldiv_loss
    return loss, data_loss, kldiv_loss

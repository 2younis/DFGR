# https://github.com/amanchadha/coursera-gan-specialization/blob/c20539e62934775711fd4f7165dfb5653c130875/C2%20-%20Build%20Better%20Generative%20Adversarial%20Networks/Week%203/BigGAN.ipynb

import torch
import torch.nn.functional as F
from numpy.random import default_rng
from scipy.stats import truncnorm
from torch import nn


class Generator(nn.Module):
    """
    Generator Class
    Values:
    base_channels: the number of base channels, a scalar
    bottom_width: the height/width of image before it gets upsampled, a scalar
    z_dim: the dimension of random noise sampled, a scalar
    shared_dim: the dimension of shared class embeddings, a scalar
    n_classes: the number of image classes, a scalar
    """

    def __init__(self, cfg):
        super().__init__()

        self.base_channels = cfg["gen_base_channels"]
        self.bottom_width = cfg["bottom_width"]
        self.z_dim = cfg["z_dim"]
        self.shared_dim = cfg["shared_dim"]
        self.n_chunks = cfg["num_chunks"]
        self.n_classes = cfg["num_classes"]
        self.img_channels = cfg["img_channels"]

        self.device = cfg["device"]

        self.z_chunk_size = self.z_dim // self.n_chunks
        self.rng = default_rng()

        # No spectral normalization on embeddings, which authors observe to cripple the generator
        self.shared_emb = nn.Embedding(self.n_classes, self.shared_dim)

        self.proj_z = nn.Linear(
            self.z_chunk_size * 4, 8 * self.base_channels * self.bottom_width**2
        )

        # Can't use one big nn.Sequential since we are adding class+noise at each block
        self.g_blocks = nn.ModuleList(
            [
                GResidualBlock(
                    self.shared_dim + self.z_chunk_size,
                    8 * self.base_channels,
                    4 * self.base_channels,
                ),  # 4_8
                GResidualBlock(
                    self.shared_dim + self.z_chunk_size,
                    4 * self.base_channels,
                    2 * self.base_channels,
                ),  # 8_16
                GResidualBlock(
                    self.shared_dim + self.z_chunk_size,
                    2 * self.base_channels,
                    self.base_channels,
                ),  # 16_32
            ]
        )

        self.attn_block = AttentionBlock(2 * self.base_channels)
        self.proj_o = nn.Sequential(
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(
                    self.base_channels, self.img_channels, kernel_size=1, padding=0
                )
            ),
            nn.Tanh(),
        )

        self.weights_init()

    def weights_init(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Embedding, nn.Linear)):
                nn.init.orthogonal_(module.weight)

    def generate(
        self, classes, batch_size=32, labels=None, trunc=None, probabilities=None
    ):

        if labels is None:
            labels = torch.from_numpy(
                self.rng.choice(list(classes.keys()), size=batch_size, p=probabilities)
            )

        if trunc is None:
            rand_z = torch.randn((batch_size, self.z_dim), device=self.device)
        else:
            truncated_z = truncnorm.rvs(-trunc, trunc, size=(batch_size, self.z_dim))
            rand_z = torch.Tensor(truncated_z).to(self.device)

        labels_emb = self.shared_emb(labels.to(self.device))
        fakes = self.forward(rand_z, labels_emb)

        return fakes, labels

    def forward(self, z, y):
        """
        z: random noise with size self.z_dim
        y: class embeddings with size self.shared_dim
            = NOTE =
            y should be class embeddings from self.shared_emb, not the raw class labels
        """
        # Chunk z and concatenate to shared class embeddings
        zs = torch.split(z, self.z_chunk_size * 4, dim=1)
        z = zs[0]

        z_l = torch.split(zs[1], self.z_chunk_size, dim=1)
        ys = [torch.cat([y, z], dim=1) for z in z_l]

        # Project noise and reshape to feed through generator blocks
        h = self.proj_z(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Feed through generator blocks
        for idx, g_block in enumerate(self.g_blocks):
            h = g_block(h, ys[idx])
            if idx == 1:  # Attention at 16x16
                h = self.attn_block(h)

        # Project to 3 RGB channels with tanh to map values to [-1, 1]
        h = self.proj_o(h)

        return h


class AttentionBlock(nn.Module):
    """
    AttentionBlock Class
    Values:
    channels: number of channels in input
    """

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.theta = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False)
        )
        self.phi = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False)
        )
        self.g = nn.utils.spectral_norm(
            nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False)
        )
        self.o = nn.utils.spectral_norm(
            nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False)
        )

        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        spatial_size = x.shape[2] * x.shape[3]

        # Apply convolutions to get query (theta), key (phi), and value (g) transforms
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)

        # Reshape spatial size for self-attention
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)

        # Compute dot product attention with query (theta) and key (phi) matrices
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)

        # Compute scaled dot product attention with value (g) and attention (beta) matrices
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(
                -1, self.channels // 2, x.shape[2], x.shape[3]
            )
        )

        # Apply gain and residual
        return self.gamma * o + x


class GResidualBlock(nn.Module):
    """
    GResidualBlock Class
    Values:
    c_dim: the dimension of conditional vector [c, z], a scalar
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    """

    def __init__(self, c_dim, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.bn1 = ClassConditionalBatchNorm2d(c_dim, in_channels)
        self.bn2 = ClassConditionalBatchNorm2d(c_dim, out_channels)

        self.activation = nn.LeakyReLU()
        self.upsample_fn = nn.Upsample(
            scale_factor=2
        )  # upsample occurs in every gblock

        self.mixin = in_channels != out_channels
        if self.mixin:
            self.conv_mixin = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            )

    def forward(self, x, y):
        # h := upsample(x, y)
        h = self.bn1(x, y)
        h = self.activation(h)
        h = self.upsample_fn(h)
        h = self.conv1(h)

        # h := conv(h, y)
        h = self.bn2(h, y)
        h = self.activation(h)
        h = self.conv2(h)

        # x := upsample(x)
        x = self.upsample_fn(x)
        if self.mixin:
            x = self.conv_mixin(x)

        return h + x


class ClassConditionalBatchNorm2d(nn.Module):
    """
    ClassConditionalBatchNorm2d Class
    Values:
    in_channels: the dimension of the class embedding (c) + noise vector (z), a scalar
    out_channels: the dimension of the activation tensor to be normalized, a scalar
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.class_scale_transform = nn.utils.spectral_norm(
            nn.Linear(in_channels, out_channels, bias=False)
        )
        self.class_shift_transform = nn.utils.spectral_norm(
            nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, x, y):
        normalized_image = self.bn(x)
        class_scale = (1 + self.class_scale_transform(y))[:, :, None, None]
        class_shift = self.class_shift_transform(y)[:, :, None, None]
        transformed_image = class_scale * normalized_image + class_shift
        return transformed_image

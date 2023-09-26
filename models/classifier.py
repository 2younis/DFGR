import torch
from torch import nn

# https://github.com/amanchadha/coursera-gan-specialization/blob/main/C2%20-%20Build%20Better%20Generative%20Adversarial%20Networks/Week%203/BigGAN.ipynb


class Classifier(nn.Module):
    """
    Classifier Class
    Values:
    base_channels: the number of base channels, a scalar
    n_classes: the number of image classes, a scalar
    """

    def __init__(self, cfg):
        super().__init__()

        self.base_channels = cfg["cl_base_channels"]
        self.img_channels = cfg["img_channels"]
        self.n_classes = cfg["num_classes"]

        self.blocks = nn.Sequential(
            ResidualBlock(
                self.img_channels,
                self.base_channels,
                downsample=False,
                use_preactivation=False,
            ),
            ResidualBlock(
                self.base_channels,
                2 * self.base_channels,
                downsample=True,
                use_preactivation=True,
            ),
            ResidualBlock(
                2 * self.base_channels,
                4 * self.base_channels,
                downsample=True,
                use_preactivation=True,
            ),
            ResidualBlock(
                4 * self.base_channels,
                8 * self.base_channels,
                downsample=True,
                use_preactivation=True,
            ),
            ResidualBlock(
                8 * self.base_channels,
                16 * self.base_channels,
                downsample=True,
                use_preactivation=True,
            ),
            nn.LeakyReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16 * self.base_channels, self.n_classes)

        self.weights_init()
        self.print_param()

    def weights_init(self):

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

    def print_param(self):
        params = 0
        for module in self.modules():
            params += sum([p.data.nelement() for p in module.parameters()])

        print(
            "Classifier's trainable parameters:  {:.0f}M {:.0f}K {:d}".format(
                params // 1e6, (params // 1e3) % 1000, params % 1000
            )
        )

        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print("Model Classifier's size: {:.3f}MB".format(size_all_mb))

    def register_hooks(self, Hook):
        self.hooks = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.hooks.append(Hook(module))

    def forward(self, x):
        h = self.blocks(x)

        h = self.avgpool(h)

        h = h.view(h.shape[0], -1)
        out = self.fc(h)

        return out, h


class ResidualBlock(nn.Module):
    """
    ResidualBlock Class
    Values:
    in_channels: the number of channels in the input, a scalar
    out_channels: the number of channels in the output, a scalar
    downsample: whether to apply downsampling
    use_preactivation: whether to apply an activation function before the first convolution
    """

    def __init__(
        self, in_channels, out_channels, downsample=True, use_preactivation=False
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.LeakyReLU()
        self.use_preactivation = (
            use_preactivation  # apply preactivation in all except first dblock
        )

        self.downsample = downsample  # downsample occurs in all except last dblock
        if downsample:
            self.downsample_fn = nn.AvgPool2d(2)
        self.mixin = (in_channels != out_channels) or downsample
        if self.mixin:
            self.conv_mixin = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def _residual(self, x):
        if self.use_preactivation:
            if self.mixin:
                x = self.conv_mixin(x)
            if self.downsample:
                x = self.downsample_fn(x)
        else:
            if self.downsample:
                x = self.downsample_fn(x)
            if self.mixin:
                x = self.conv_mixin(x)
        return x

    def forward(self, x):
        # Apply preactivation if applicable
        if self.use_preactivation:
            h = self.activation(self.bn1(x))
        else:
            h = x

        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        if self.downsample:
            h = self.downsample_fn(h)

        return h + self._residual(x)

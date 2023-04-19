import torch
from torch import nn

# https://github.com/amanchadha/coursera-gan-specialization/blob/c20539e62934775711fd4f7165dfb5653c130875/C2%20-%20Build%20Better%20Generative%20Adversarial%20Networks/Week%203/BigGAN.ipynb


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

    def weights_init(self):

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

    def register_hooks(self):
        self.hooks = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.hooks.append(FeatureMeanVarHook(module))

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


# https://github.com/zju-vipa/MosaicKD/blob/33e33640f1528e38070b07b1f729e9fe459b27cd/engine/hooks.py


class FeatureMeanVarHook:
    def __init__(self, module, on_input=True, dim=[0, 2, 3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim
        self.var, self.mean = None, None

    def hook_fn(self, module, input, output):
        # To avoid inplace modification

        if self.on_input:
            feature = input[0].clone()
        else:
            feature = output.clone()
        self.var, self.mean = torch.var_mean(feature, dim=self.dim, unbiased=True)

    def remove(self):
        self.hook.remove()

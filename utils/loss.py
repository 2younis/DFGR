import math
import random

import torch
import torch.nn.functional as F
from torch import nn


# https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL/blob/main/learners/datafree_helper.py#L188
class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma=1, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def js_divergence(imgs, num_samples):
    batch_size = imgs.size(0)

    sample1_indices = random.sample(range(0, batch_size // 2), num_samples)
    sample2_indices = random.sample(range(batch_size // 2, batch_size), num_samples)

    imgs = imgs.view([batch_size, -1])
    sample1 = imgs[sample1_indices]
    sample2 = imgs[sample2_indices]

    sample1 = F.log_softmax(sample1, dim=1)
    sample2 = F.log_softmax(sample2, dim=1)

    js_div = (
        F.kl_div(sample1, sample2, reduction="batchmean", log_target=True)
        + F.kl_div(sample2, sample1, reduction="batchmean", log_target=True)
    ) / 2

    return js_div


# https://math.stackexchange.com/q/453794
def merge_gaussians(features_dict, labels):
    means = features_dict["mean"]
    var = features_dict["var"]

    bin_count = torch.bincount(labels).tolist()
    bin_count = [i for i in bin_count if i != 0]

    mu = sum(means[i] * count for i, count in enumerate(bin_count)) / sum(bin_count)

    sigma = sum(
        (count * (var[i] + (means[i] ** 2))) for i, count in enumerate(bin_count)
    ) / sum(bin_count)

    sigma -= mu**2

    return sigma, mu


def focal_loss(logits, targets, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (((1 - pt) ** gamma) * ce_loss).mean()

    return loss


# https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/regularization.py#L61


def distillation_loss(output, previous_output, temperature=2.0):

    log_p = torch.log_softmax(output / temperature, dim=1)
    q = torch.softmax(previous_output / temperature, dim=1)

    dist_loss = F.kl_div(log_p, q, reduction="batchmean")

    return dist_loss

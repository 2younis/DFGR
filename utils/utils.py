import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.dataset import TrainDatasetUnbalanced


# https://github.com/NVlabs/DeepInversion/blob/master/deepinversion.py#L29
class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2
        )

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


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
        super().__init__()
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


def cumulative(lists):

    return [sum(lists[0:x:1]) for x in range(1, len(lists) + 1)]


def initialize_model_and_optimizer(cfg, model_class):
    model = model_class(cfg).to(cfg["device"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer_lr"],
        betas=(cfg["optimizer_beta_1"], cfg["optimizer_beta_2"]),
    )
    return model, optimizer


def save_features(cfg):
    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

    cfg["classifier"].eval()

    dataset = TrainDatasetUnbalanced(cfg, trained_tasks, dataset=cfg["dataset"])
    dataloader = DataLoader(dataset, batch_size=cfg["val_batch_size"], shuffle=True)

    features = None
    classes = None

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])
            _, h = cfg["classifier"](imgs)

            if features is None:
                features = h.detach().cpu()
                classes = labels.detach().cpu()
            else:
                features = torch.cat([features, h.detach().cpu()], dim=0)
                classes = torch.cat([classes, labels.detach().cpu()], dim=0)

    values, indices = torch.sort(classes)
    bin_count = torch.bincount(classes).tolist()

    bin_count = [i for i in bin_count if i != 0]
    bin_count.insert(0, 0)
    bin_count_cum = cumulative(bin_count)

    clss = [int(classes[indices[bin_count_cum[a]]]) for a in range(len(bin_count) - 1)]

    mean = torch.empty((len(clss), features.shape[1]))
    var = torch.empty((len(clss), features.shape[1]))

    for a in range(len(bin_count) - 1):
        mean[a] = torch.mean(
            features[indices[bin_count_cum[a] : bin_count_cum[a + 1]]], dim=0
        )

        var[a] = torch.var(
            features[indices[bin_count_cum[a] : bin_count_cum[a + 1]]], dim=0
        )

    features_dict = {"mean": mean, "var": var, "labels": clss}
    torch.save(features_dict, cfg["features_file"])


def adjust_replay_probabilities(loss, labels, p):
    replay_labels = list(set(labels.tolist()))

    if p is None:
        len_labels = len(replay_labels)
        p = [1.0 / len_labels] * len_labels
        return p

    _, indices = torch.sort(labels)
    bin_count = torch.bincount(labels).tolist()
    bin_count = [i for i in bin_count if i != 0]
    bin_count.insert(0, 0)
    bin_count_cum = cumulative(bin_count)

    avg_losses = [
        torch.mean(loss[indices[bin_count_cum[a] : bin_count_cum[a + 1]]]).item()
        for a in range(len(bin_count) - 1)
    ]
    normalized_avg_losses = [a / bin_count[i + 1] for i, a in enumerate(avg_losses)]

    p = [(x * y) for x, y in zip(p, normalized_avg_losses)]
    mean_p = sum(p) / len(p)
    p = [a + mean_p for a in p]
    p = [a / sum(p) for a in p]

    return p


def get_fisher_diag(cfg, trained_tasks, params):

    fisher = {}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        fisher[n] = Variable(p.data)

    dataset = TrainDatasetUnbalanced(cfg, trained_tasks, dataset=cfg["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=False
    )
    dataset_len = len(dataset)

    cfg["classifier"].eval()

    for imgs, labels in dataloader:

        imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

        cfg["classifier"].zero_grad()
        output, _ = cfg["classifier"](imgs)

        loss = F.cross_entropy(output, labels)
        loss.backward()

        for n, p in cfg["classifier"].named_parameters():

            if p.grad is not None:
                fisher[n].data += (p.grad.data**2) / dataset_len

    return fisher


def apply_prunning(cfg, task, trained_tasks):

    total_tasks = task | trained_tasks

    for classe in range(cfg["num_classes"]):
        if classe not in total_tasks:
            cfg["classifier"].fc.weight.data[classe] *= 0.0
            cfg["classifier"].fc.bias.data[classe] *= 0.0

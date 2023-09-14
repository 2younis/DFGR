import math
import random

import torch
import torch.nn.functional as F
from torch import nn


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


def distillation_loss(output, previous_output, trained_tasks, temperature=2.0):

    previous_classes = list(trained_tasks)

    log_p = torch.log_softmax(output[:, previous_classes] / temperature, dim=1)
    q = torch.softmax(previous_output[:, previous_classes] / temperature, dim=1)

    dist_loss = F.kl_div(log_p, q, reduction="batchmean")

    return dist_loss

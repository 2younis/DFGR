import random

import torch
import torch.nn.functional as F


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


def merge_gaussians(features_dict, labels):
    means = features_dict["mean"]
    var = features_dict["var"]

    bin_count = torch.bincount(labels).tolist()
    bin_count = [i for i in bin_count if i != 0]

    mu = sum(means[i] * count for i, count in enumerate(bin_count)) / sum(bin_count)

    sigma = sum(
        (count * (var[i] + ((means[i] - mu) ** 2))) for i, count in enumerate(bin_count)
    ) / sum(bin_count)

    return sigma, mu


def focal_loss(cfg, outputs, targets, gamma=2.0):

    bin_count = torch.bincount(targets)
    freq = bin_count / sum(bin_count)

    weight = torch.div(1, freq)
    weight[weight == float("inf")] = 10.0

    weight = torch.cat(
        (
            weight,
            10.0 * torch.ones(cfg["num_classes"] - len(weight), device=cfg["device"]),
        )
    )

    ce_loss = F.cross_entropy(outputs, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (((1 - pt) ** gamma) * ce_loss).mean()

    return loss

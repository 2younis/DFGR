from copy import deepcopy

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataset import TrainDatasetUnbalanced


def cumulative(lists):

    return [sum(lists[0:x:1]) for x in range(1, len(lists) + 1)]


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

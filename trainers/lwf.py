from copy import deepcopy

import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import utils
import losses as ls
from dataset import TrainDatasetUnbalanced


def train_classifier_lwf(cfg, task):

    trained_tasks = {}
    previous_model = None

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

        previous_model = deepcopy(cfg["classifier"])
        previous_model.to(cfg["device"])

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    lwf_alpha = 0.1
    temperature = 2.0

    if trained_tasks:

        weight = cfg["classifier"].fc.weight.data.clone()
        bias = cfg["classifier"].fc.bias.data.clone()

        nn.init.xavier_uniform_(cfg["classifier"].fc.weight)
        cfg["classifier"].fc.bias.data.fill_(0)

        utils.apply_prunning(cfg, task, trained_tasks)

        for classe in range(cfg["num_classes"]):
            if classe in trained_tasks:
                cfg["classifier"].fc.weight.data[classe] = weight[classe]
                cfg["classifier"].fc.bias.data[classe] = bias[classe]

    cfg["classifier"].train()

    dataset = TrainDatasetUnbalanced(cfg, task, dataset=cfg["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=True
    )

    mlflow.log_param("classifier trained tasks", trained_tasks)
    mlflow.log_param("lwf_alpha", lwf_alpha)
    mlflow.log_param("temperature", temperature)

    dataset_len = len(dataset)

    for epoch in range(cfg["max_epochs"]):
        orginal_losses = 0
        dist_losses = 0
        train_losses = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            utils.apply_prunning(cfg, task, trained_tasks)

            cfg["cl_optimizer"].zero_grad()
            output, _ = cfg["classifier"](imgs)

            orginal_loss = F.cross_entropy(output, labels)

            if trained_tasks:
                previous_output, _ = previous_model(imgs)

                dist_loss = ls.distillation_loss(output, previous_output, temperature)

                train_loss = orginal_loss + (lwf_alpha * dist_loss)
                dist_losses += dist_loss.item()

            else:
                train_loss = orginal_loss

            train_loss.backward()
            cfg["cl_optimizer"].step()

            orginal_losses += orginal_loss.item()
            train_losses += train_loss.item()

        orginal_losses /= dataset_len / cfg["cl_batch_size"]
        dist_losses /= dataset_len / cfg["cl_batch_size"]
        train_losses /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_losses, step=epoch)
        mlflow.log_metric("Original Loss", orginal_losses, step=epoch)
        mlflow.log_metric("Distilation Loss", dist_losses, step=epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            model_checkpoint = {
                "model": cfg["classifier"].state_dict(),
                "trained_tasks": task | trained_tasks,
                "batch_size": cfg["cl_batch_size"],
            }
            torch.save(model_checkpoint, cfg["classifier_model_file"])
        else:
            patience_counter += 1

        mlflow.log_metric("Best loss", best_loss, step=epoch)
        mlflow.log_metric("Patience Counter", patience_counter, step=epoch)

        if patience_counter >= cfg["cl_max_patience"]:
            break

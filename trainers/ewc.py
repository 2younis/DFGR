from copy import deepcopy

import mlflow
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from dataset import TrainDatasetUnbalanced


def train_classifier_ewc(cfg, task):

    trained_tasks = {}

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    if not trained_tasks:
        ewc_lambda = 0
    else:
        ewc_lambda = 1000

        params = {
            n: p for n, p in cfg["classifier"].named_parameters() if p.requires_grad
        }

        fisher_matrix = utils.get_fisher_diag(cfg, trained_tasks, params)

        param_old = {}
        for n, p in deepcopy(params).items():
            param_old[n] = Variable(p.data)

    cfg["classifier"].train()

    dataset = TrainDatasetUnbalanced(cfg, task, dataset=cfg["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=True
    )

    mlflow.log_param("classifier trained tasks", trained_tasks)
    mlflow.log_param("ewc_lambda", ewc_lambda)

    dataset_len = len(dataset)

    for epoch in range(cfg["max_epochs"]):
        orginal_losses = 0
        ewc_losses = 0
        train_losses = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            cfg["cl_optimizer"].zero_grad()
            output, _ = cfg["classifier"](imgs)

            orginal_loss = F.cross_entropy(output, labels)

            if trained_tasks:
                ewc_loss = 0
                for n, p in cfg["classifier"].named_parameters():
                    _loss = fisher_matrix[n] * ((p - param_old[n]) ** 2)
                    ewc_loss += _loss.sum()

                train_loss = orginal_loss + (ewc_lambda * ewc_loss)
                ewc_losses += ewc_loss.item()

            else:
                train_loss = orginal_loss

            train_loss.backward()
            cfg["cl_optimizer"].step()

            orginal_losses += orginal_loss.item()
            train_losses += train_loss.item()

        orginal_losses /= dataset_len / cfg["cl_batch_size"]
        ewc_losses /= dataset_len / cfg["cl_batch_size"]
        train_losses /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_losses, step=epoch)
        mlflow.log_metric("Original Loss", orginal_losses, step=epoch)
        mlflow.log_metric("EWC Loss", ewc_losses, step=epoch)

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

import mlflow
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import TrainDatasetUnbalanced


def train_classifier(cfg, task):

    trained_tasks = {}

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    cfg["classifier"].train()

    dataset = TrainDatasetUnbalanced(cfg, task, dataset=cfg["dataset"])
    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=True
    )
    dataset_len = len(dataset)

    mlflow.log_param("classifier trained tasks", trained_tasks)

    for epoch in range(cfg["max_epochs"]):
        train_losses = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            cfg["cl_optimizer"].zero_grad()
            output, _ = cfg["classifier"](imgs)

            train_loss = F.cross_entropy(output, labels)

            train_loss.backward()
            cfg["cl_optimizer"].step()
            train_losses += train_loss.item()

        train_losses /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_losses, step=epoch)

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

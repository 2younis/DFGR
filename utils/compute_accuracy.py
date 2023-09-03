import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import (
    TestDatasetPartial,
    TrainDatasetPartial,
)


def validate_classifier(cfg):
    trained_tasks = {}

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

    cfg["classifier"].eval()

    mlflow.log_param("validation task", trained_tasks)

    overall_correct = 0
    overall_total = 0

    for task_id, _ in trained_tasks.items():
        dataset = TrainDatasetPartial(cfg, task_id, dataset=cfg["dataset"])
        dataloader = DataLoader(dataset, batch_size=cfg["val_batch_size"])

        total = len(dataset)
        overall_total += total

        val_loss = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])
                output, _ = cfg["classifier"](imgs)

                val_loss += F.cross_entropy(output, labels).item()
                max_indices = output.max(1)[1]
                correct += (max_indices == labels).sum().detach().item()

                val_loss /= total / cfg["val_batch_size"]

        overall_correct += correct

        mlflow.log_metric("val loss task id " + str(task_id), val_loss)
        mlflow.log_metric(
            "val accuracy task id " + str(task_id), 100.0 * correct / total
        )

    mlflow.log_metric("val overall accuracy", 100.0 * overall_correct / overall_total)


def test_classifier(cfg):
    trained_tasks = {}

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

    cfg["classifier"].eval()

    mlflow.log_param("test task", trained_tasks)

    overall_correct = 0
    overall_total = 0

    for task_id, _ in trained_tasks.items():
        dataset = TestDatasetPartial(cfg, task_id, dataset=cfg["dataset"])
        dataloader = DataLoader(dataset, batch_size=cfg["val_batch_size"])

        total = len(dataset)
        overall_total += total

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for imgs, labels in dataloader:

                imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])
                output, _ = cfg["classifier"](imgs)

                test_loss += F.cross_entropy(output, labels).item()
                max_indices = output.max(1)[1]
                correct += (max_indices == labels).sum().detach().item()

                test_loss /= total / cfg["val_batch_size"]

        overall_correct += correct

        mlflow.log_metric("test loss task id " + str(task_id), test_loss)
        mlflow.log_metric(
            "test accuracy task id " + str(task_id), 100.0 * correct / total
        )

    mlflow.log_metric("test overall accuracy", 100.0 * overall_correct / overall_total)

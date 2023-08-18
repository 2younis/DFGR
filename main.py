import copy
import os
import shutil

import mlflow
import torch
import trainer as tr
from configs import config
from models.classifier import Classifier
from models.generator import Generator
from utils.prepare_dataset import create_dataset

cfg = config.cfg("configs/config.yaml")


def train_dgr(image_dataset):

    cfg["generator"] = Generator(cfg).to(cfg["device"])
    cfg["g_optimizer"] = torch.optim.Adam(
        cfg["generator"].parameters(),
        lr=cfg["optimizer_lr"],
        betas=(cfg["optimizer_beta_1"], cfg["optimizer_beta_2"]),
    )

    cfg["classifier"] = Classifier(cfg).to(cfg["device"])
    cfg["cl_optimizer"] = torch.optim.Adam(
        cfg["classifier"].parameters(),
        lr=cfg["optimizer_lr"],
        betas=(cfg["optimizer_beta_1"], cfg["optimizer_beta_2"]),
    )

    cfg["dataset"] = image_dataset

    for i, training_params in enumerate(cfg["training_params"]):
        gen_params, adjust_replay = training_params

        for scenario_name, scenario_tasks in cfg["scenarios"].items():
            tasks = copy.deepcopy(scenario_tasks)

            cfg["generator"].weights_init()
            cfg["classifier"].weights_init()

            os.makedirs("saved_models", exist_ok=True)
            mlflow.set_experiment(f"case_{i} {scenario_name} {image_dataset}")

            for task_no, cl_task in enumerate(tasks):
                with mlflow.start_run():
                    mlflow.log_param("task_no", task_no)
                    mlflow.log_param("task", cl_task)
                    tr.train_classifier(cfg, cl_task, adjust_replay)

                with mlflow.start_run():
                    tr.validate_classifier(cfg)
                    tr.test_classifier(cfg)

                if task_no < (cfg["total_tasks"] - 1):
                    with mlflow.start_run():
                        mlflow.log_param("task_no", task_no)
                        tr.train_generator(cfg, cl_task, gen_params)

            shutil.rmtree("saved_models/")

    del cfg["generator"]
    del cfg["g_optimizer"]
    del cfg["classifier"]
    del cfg["cl_optimizer"]


def train_baselines(image_dataset, strategy):

    cfg["classifier"] = Classifier(cfg).to(cfg["device"])
    cfg["cl_optimizer"] = torch.optim.Adam(
        cfg["classifier"].parameters(),
        lr=cfg["optimizer_lr"],
        betas=(cfg["optimizer_beta_1"], cfg["optimizer_beta_2"]),
    )

    cfg["dataset"] = image_dataset

    for scenario_name, scenario_tasks in cfg["scenarios"].items():
        tasks = copy.deepcopy(scenario_tasks)

        cfg["classifier"].weights_init()

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(f"{strategy} {scenario_name} {image_dataset}")

        for task_no, cl_task in enumerate(tasks):
            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                mlflow.log_param("task", cl_task)

                if strategy == "EWC":
                    tr.train_classifier_ewc(cfg, cl_task)
                elif strategy == "LWF":
                    tr.train_classifier_lwf(cfg, cl_task)
                elif strategy == "Naive":
                    tr.train_classifier_naive(cfg, cl_task)

            with mlflow.start_run():
                tr.validate_classifier(cfg)
                tr.test_classifier(cfg)

        shutil.rmtree("saved_models/")

    del cfg["classifier"]
    del cfg["cl_optimizer"]


if __name__ == "__main__":

    create_dataset()

    for image_dataset in cfg["image_datasets"]:
        train_dgr(image_dataset)

        for strategy in cfg["baseline_strategies"]:
            train_baselines(image_dataset, strategy)

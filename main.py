import copy
import os
import shutil

import mlflow
import torch
from configs import config
from models.classifier import Classifier
from models.generator import Generator
from trainer import (
    test_classifier,
    train_classifier,
    train_classifier_ewc,
    train_generator,
    validate_classifier,
)
from utils.prepare_dataset import create_dataset

cfg = config.cfg("configs/config.yaml")

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


def train_cl(image_dataset):

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
                    train_classifier(cfg, cl_task, adjust_replay)

                with mlflow.start_run():
                    validate_classifier(cfg)
                    test_classifier(cfg)

                if task_no < (cfg["total_tasks"] - 1):
                    with mlflow.start_run():
                        mlflow.log_param("task_no", task_no)
                        train_generator(cfg, cl_task, gen_params)

            shutil.rmtree("saved_models/")


def train_ewc(image_dataset):

    cfg["dataset"] = image_dataset

    for scenario_name, scenario_tasks in cfg["scenarios"].items():
        tasks = copy.deepcopy(scenario_tasks)

        cfg["generator"].weights_init()
        cfg["classifier"].weights_init()

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(f"EWC {scenario_name} {image_dataset}")

        for task_no, cl_task in enumerate(tasks):
            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                mlflow.log_param("task", cl_task)
                train_classifier_ewc(cfg, cl_task)

            with mlflow.start_run():
                validate_classifier(cfg)
                test_classifier(cfg)

        shutil.rmtree("saved_models/")


if __name__ == "__main__":

    create_dataset()

    for image_dataset in cfg["image_datasets"]:
        train_cl(image_dataset)
        train_ewc(image_dataset)

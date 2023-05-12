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
    train_generator,
    validate_classifier,
)

cfg = config.cfg("configs/config.yaml")

cfg["generator"] = Generator(cfg).to(cfg["device"])
cfg["g_optimizer"] = torch.optim.Adam(
    cfg["generator"].parameters(),
    lr=cfg["optimizer_lr"],
    betas=(cfg["gen_optimizer_beta_1"], cfg["optimizer_beta_2"]),
)

cfg["classifier"] = Classifier(cfg).to(cfg["device"])
cfg["cl_optimizer"] = torch.optim.Adam(
    cfg["classifier"].parameters(),
    lr=cfg["optimizer_lr"],
    betas=(cfg["cl_optimizer_beta_1"], cfg["optimizer_beta_2"]),
)

for i, training_params in enumerate(cfg["training_params"]):
    gen_params, adjust_replay = training_params

    for scenario_name, scenario_tasks in cfg["scenarios"].items():
        tasks = copy.deepcopy(scenario_tasks)

        cfg["generator"].weights_init()
        cfg["classifier"].weights_init()

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(scenario_name + " " + str(i))

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

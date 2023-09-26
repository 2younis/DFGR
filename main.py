import copy
import os
import shutil

import mlflow
from configs import config
from models import Classifier, Generator
from trainers import *
from utils import *

cfg = config("configs/config.yaml")


def train_dfgr(image_dataset):

    cfg["classifier"], cfg["cl_optimizer"] = initialize_model_and_optimizer(
        cfg, Classifier
    )
    cfg["generator"], cfg["g_optimizer"] = initialize_model_and_optimizer(
        cfg, Generator
    )

    cfg["dataset"] = image_dataset

    for i, training_params in enumerate(cfg["training_params"]):
        gen_params, adjust_replay = training_params

        for scenario_name, scenario_tasks in cfg["scenarios"].items():
            tasks = copy.deepcopy(scenario_tasks)

            cfg["classifier"].weights_init()
            cfg["generator"].weights_init()

            os.makedirs("saved_models", exist_ok=True)
            mlflow.set_experiment(f"yo_case_{i} {scenario_name} {image_dataset}")

            for task_no, cl_task in enumerate(tasks):
                with mlflow.start_run():
                    mlflow.log_param("task_no", task_no)
                    mlflow.log_param("task", cl_task)
                    dfgr.train_classifier(cfg, cl_task, adjust_replay)

                with mlflow.start_run():
                    validate_classifier(cfg)
                    test_classifier(cfg)

                if task_no < (cfg["total_tasks"] - 1):
                    with mlflow.start_run():
                        mlflow.log_param("task_no", task_no)
                        dfgr.train_generator(cfg, cl_task, gen_params)

            shutil.rmtree("saved_models/")

    del cfg["classifier"]
    del cfg["cl_optimizer"]
    del cfg["generator"]
    del cfg["g_optimizer"]


def train_baselines(image_dataset, strategy):

    cfg["classifier"], cfg["cl_optimizer"] = initialize_model_and_optimizer(
        cfg, Classifier
    )

    cfg["dataset"] = image_dataset

    for scenario_name, scenario_tasks in cfg["scenarios"].items():
        tasks = copy.deepcopy(scenario_tasks)

        cfg["classifier"].weights_init()

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(f"yo_{strategy} {scenario_name} {image_dataset}")

        for task_no, cl_task in enumerate(tasks):
            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                mlflow.log_param("task", cl_task)

                eval(strategy.lower()).train_classifier(cfg, cl_task)

            with mlflow.start_run():
                validate_classifier(cfg)
                test_classifier(cfg)

        shutil.rmtree("saved_models/")

    del cfg["classifier"]
    del cfg["cl_optimizer"]


if __name__ == "__main__":

    create_dataset()

    for image_dataset in cfg["image_datasets"]:
        train_dfgr(image_dataset)

        for strategy in cfg["baseline_strategies"]:
            train_baselines(image_dataset, strategy)

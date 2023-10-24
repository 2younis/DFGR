import copy
import os
import shutil

import mlflow
import utils
from configs import config
from models import Classifier, Generator
from trainers import *

cfg = config("configs/config.yaml")


def train_dfgr(image_dataset):

    cfg["classifier"], cfg["cl_optimizer"] = utils.initialize_model_and_optimizer(
        cfg, Classifier
    )
    cfg["generator"], cfg["g_optimizer"] = utils.initialize_model_and_optimizer(
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
                    utils.validate_classifier(cfg)
                    utils.test_classifier(cfg)

                if task_no < (cfg["total_tasks"] - 1):
                    with mlflow.start_run():
                        mlflow.log_param("task_no", task_no)
                        dfgr.train_generator(cfg, cl_task, gen_params)

            shutil.rmtree("saved_models/")

    del cfg["classifier"]
    del cfg["cl_optimizer"]
    del cfg["generator"]
    del cfg["g_optimizer"]


def train_baselines(image_dataset, strategy, advanced_strategy=False):

    if strategy in cfg["advanced_strategies"]:
        advanced_strategy = True

    cfg["classifier"], cfg["cl_optimizer"] = utils.initialize_model_and_optimizer(
        cfg, Classifier
    )
    if advanced_strategy:
        cfg["generator"], cfg["g_optimizer"] = utils.initialize_model_and_optimizer(
            cfg, Generator
        )

    cfg["dataset"] = image_dataset

    for scenario_name, scenario_tasks in cfg["scenarios"].items():
        tasks = copy.deepcopy(scenario_tasks)

        cfg["classifier"].weights_init()
        if advanced_strategy:
            cfg["generator"].weights_init()

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(f"{strategy} {scenario_name} {image_dataset}")

        for task_no, cl_task in enumerate(tasks):
            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                mlflow.log_param("task", cl_task)
                eval(strategy.lower()).train_classifier(cfg, cl_task)

            with mlflow.start_run():
                utils.validate_classifier(cfg)
                utils.test_classifier(cfg)

            if advanced_strategy and task_no < (cfg["total_tasks"] - 1):
                with mlflow.start_run():
                    mlflow.log_param("task_no", task_no)
                    eval(strategy.lower()).train_generator(cfg, cl_task)

        shutil.rmtree("saved_models/")

    del cfg["classifier"]
    del cfg["cl_optimizer"]
    if advanced_strategy:
        del cfg["generator"]
        del cfg["g_optimizer"]


if __name__ == "__main__":

    utils.create_dataset()

    for image_dataset in cfg["image_datasets"]:
        train_dfgr(image_dataset)

        baseline_strategies = cfg["basic_strategies"] + cfg["advanced_strategies"]

        for strategy in baseline_strategies:
            train_baselines(image_dataset, strategy)

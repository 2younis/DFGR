import os
import shutil

import mlflow
from configs import config
from trainer import train_classifier, train_generator, validate_classifier

cfg = config.cfg("configs/config.yaml")


for i, generator_params in enumerate(cfg["generator_loss_coeffs"]):

    cfg = config.cfg("configs/config.yaml")

    for scenario_name, tasks in cfg["scenarios"].items():

        os.makedirs("saved_models", exist_ok=True)
        mlflow.set_experiment(scenario_name + " " + str(i))
        for task_no, cl_task in enumerate(tasks):

            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                mlflow.log_param("task", cl_task)

                train_classifier(cl_task)

            with mlflow.start_run():
                validate_classifier()

            with mlflow.start_run():
                mlflow.log_param("task_no", task_no)
                train_generator(cl_task, generator_params)

        shutil.rmtree("saved_models/")

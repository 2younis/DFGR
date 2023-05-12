from pathlib import Path

import torch
import yaml


def cfg(path):
    with open(path, "r") as file:
        try:
            config = yaml.safe_load(file)

            config["device"] = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            config["generator_checkpoint_path"] = Path(config["generator_model_file"])
            config["classifier_checkpoint_path"] = Path(config["classifier_model_file"])

            config["training_params"] = [
                ((delta, alpha, beta, gamma), adjust_replay)
                for delta in config["delta"]
                for alpha in config["alpha"]
                for beta in config["beta"]
                for gamma in config["gamma"]
                for adjust_replay in config["adjust_replay"]
                if delta == 1 or alpha == 1 or beta == 1
            ]

            config["class_loss"] = torch.zeros(
                [1], dtype=torch.float, device=config["device"]
            )
            config["features_loss"] = config["class_loss"].clone()
            config["batchmorm_loss"] = config["class_loss"].clone()
            config["div_loss"] = config["class_loss"].clone()

            config["num_div_samples"] = config["gen_batch_size"] // 3
        except yaml.YAMLError as exception:
            print(exception)

    return config


if __name__ == "__main__":

    import pprint

    config = cfg("config.yaml")
    pprint.pprint(config, sort_dicts=False)

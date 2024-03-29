from pathlib import Path

import torch
import yaml


def config(path):
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
                (
                    (delta, alpha, config["beta"], config["gamma"], config["epsilon"]),
                    adjust_replay,
                )
                for adjust_replay in config["adjust_replay"]
                for delta in config["delta"]
                for alpha in config["alpha"]
            ]

            config["class_loss"] = torch.zeros(
                [1], dtype=torch.float, device=config["device"]
            )
            config["features_loss"] = config["class_loss"].clone()
            config["batchmorm_loss"] = config["class_loss"].clone()
            config["div_loss"] = config["class_loss"].clone()
            config["smoothing_loss"] = config["class_loss"].clone()

            config["num_div_samples"] = config["gen_batch_size"] // 3
        except yaml.YAMLError as exception:
            print(exception)

    return config


if __name__ == "__main__":

    import pprint

    cfg = config("config.yaml")
    pprint.pprint(cfg, sort_dicts=False)

    print("training params combinations", len(cfg["training_params"]))

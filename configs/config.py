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
        except yaml.YAMLError as exception:
            print(exception)

    return config


if __name__ == "__main__":

    config = cfg()

    print(config)

import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

mnist_normalize = transforms.Compose(
    [
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def load_mnist(cfg):
    with open(cfg["mnist_path"], "rb") as file:
        mnist = pickle.load(file)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


def load_fashionmnist(cfg):
    with open(cfg["fashionmnist_path"], "rb") as file:
        fashionmnist = pickle.load(file)
    return (
        fashionmnist["training_images"],
        fashionmnist["training_labels"],
        fashionmnist["test_images"],
        fashionmnist["test_labels"],
    )


def get_dataset_unbalanced(imgs, labels, classes_dict=None):

    if classes_dict is not None:
        total_indices = None
        for classe, ratio in classes_dict.items():
            if total_indices is None:
                indices = np.where(labels == classe)[0]
                if ratio < 1.0:
                    num_indices = int(len(indices) * ratio)
                    indices = indices[:num_indices]
                total_indices = indices
            else:
                indices = np.where(labels == classe)[0]
                if ratio < 1.0:
                    num_indices = int(len(indices) * ratio)
                    indices = indices[:num_indices]
                total_indices = np.concatenate([total_indices, indices])

        imgs_selected = imgs[total_indices]
        labels_selected = labels[total_indices]

        return imgs_selected, labels_selected

    return None


class TrainDatasetUnbalanced(Dataset):
    def __init__(self, cfg, classes_dict, dataset="mnist"):

        self.dataset = dataset

        self.image_size = cfg["image_size"]

        if self.dataset == "mnist":
            x_train, t_train, _, _ = load_mnist(cfg)
        elif self.dataset == "fashion":
            x_train, t_train, _, _ = load_fashionmnist(cfg)
        else:
            print(f"Dataset {self.dataset} not available!")
            sys.exit()

        self.x_train, self.t_train = get_dataset_unbalanced(
            x_train, t_train, classes_dict
        )

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):

        label = int(self.t_train[idx])

        image = torch.from_numpy(self.x_train[idx])
        image = F.interpolate(image.unsqueeze(dim=0), size=self.image_size).squeeze(
            dim=0
        )
        image = mnist_normalize(image)

        return image, label


class TrainDatasetComplete(Dataset):
    def __init__(self, cfg, dataset="mnist"):

        self.image_size = cfg["image_size"]

        self.dataset = dataset

        if self.dataset == "mnist":
            self.x_train, self.t_train, _, _ = load_mnist(cfg)
        elif self.dataset == "fashion":
            self.x_train, self.t_train, _, _ = load_fashionmnist(cfg)
        else:
            print(f"Dataset {self.dataset} not available!")
            sys.exit()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):

        label = int(self.t_train[idx])

        image = torch.from_numpy(self.x_train[idx])
        image = F.interpolate(image.unsqueeze(dim=0), size=self.image_size).squeeze(
            dim=0
        )
        image = mnist_normalize(image)

        return image, label


class TestDatasetComplete(Dataset):
    def __init__(self, cfg, dataset="mnist"):

        self.image_size = cfg["image_size"]

        self.dataset = dataset

        if self.dataset == "mnist":
            _, _, self.x_test, self.t_test = load_mnist(cfg)
        elif self.dataset == "fashion":
            _, _, self.x_test, self.t_test = load_fashionmnist(cfg)
        else:
            print(f"Dataset {self.dataset} not available!")
            sys.exit()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):

        label = int(self.t_test[idx])

        image = torch.from_numpy(self.x_test[idx])
        image = F.interpolate(image.unsqueeze(dim=0), size=self.image_size).squeeze(
            dim=0
        )
        image = mnist_normalize(image)

        return image, label

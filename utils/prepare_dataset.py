import gzip
import os
import pickle
import shutil
from os import path

import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]


def init_mnist():

    if path.exists("data/mnist.pkl"):
        print("MNIST files already downloaded!")
    else:
        MNIST(path.join("data", "mnist"), download=True)
        save_mnist()
        shutil.rmtree("data/mnist")


def init_fashionmnist():

    if path.exists("data/fashionmnist.pkl"):
        print("FashionMNIST files already downloaded!")
    else:
        FashionMNIST(path.join("data", "mnist"), download=True)
        save_fashionmnist()
        shutil.rmtree("data/mnist")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open("data/mnist/MNIST/raw/" + name[1], "rb") as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    for name in filename[-2:]:
        with gzip.open("data/mnist/MNIST/raw/" + name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("data/mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def save_fashionmnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open("data/mnist/FashionMNIST/raw/" + name[1], "rb") as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            mnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    for name in filename[-2:]:
        with gzip.open("data/mnist/FashionMNIST/raw/" + name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("data/fashionmnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def create_dataset():
    os.makedirs("data", exist_ok=True)

    init_mnist()
    init_fashionmnist()


if __name__ == "__main__":

    create_dataset()

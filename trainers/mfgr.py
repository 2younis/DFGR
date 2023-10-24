from copy import deepcopy

import math
import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import utils
import losses as ls
from models import Classifier

from dataset import TrainDatasetUnbalanced

"""
Content adapted from
@article{xin2021memory,
  title={Memory-Free Generative Replay For Class-Incremental Learning},
  author={Xin, Xiaomeng and Zhong, Yiran and Hou, Yunzhong and Wang, Jinjun and Zheng, Liang},
  journal={arXiv preprint arXiv:2109.00328},
  year={2021}
}
"""
# https://github.com/xmengxin/MFGR/blob/main/run.sh
# https://github.com/xmengxin/MFGR/blob/main/train_mfgr.py

T = 2
CROSS_ENTROPY_LOSS_WEIGHT = 1
INFORMATION_ENTROPY_LOSS_WEIGHT = 5
ACTIVATION_LOSS_WEIGHT = 0.1
VARIATION_REGULARIZATION_LOSS_WEIGHT = 10
BATCHMORM_LOSS_WEIGHT = 20


def train_classifier(cfg, task):

    trained_tasks = {}
    previous_model = None
    alpha = 0

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

        previous_model, _ = utils.initialize_model_and_optimizer(cfg, Classifier)
        previous_model.load_state_dict(model_checkpoint["model"])
        previous_model.to(cfg["device"])
        previous_model.eval()

    if cfg["generator_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["generator_model_file"])
        cfg["generator"].load_state_dict(model_checkpoint["model"])
        cfg["generator"].to(cfg["device"])
        cfg["generator"].eval()

    if trained_tasks:
        alpha = len(trained_tasks) / (len(trained_tasks) + len(task))

        weight = cfg["classifier"].fc.weight.data.clone()
        bias = cfg["classifier"].fc.bias.data.clone()

        nn.init.xavier_uniform_(cfg["classifier"].fc.weight)
        cfg["classifier"].fc.bias.data.fill_(0)

        for classe in range(cfg["num_classes"]):
            if classe in trained_tasks:
                cfg["classifier"].fc.weight.data[classe] = weight[classe]
                cfg["classifier"].fc.bias.data[classe] = bias[classe]

    cfg["classifier"].train()

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    dataset = TrainDatasetUnbalanced(cfg, task, dataset=cfg["dataset"])
    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=True
    )

    mlflow.log_param("classifier previous tasks", trained_tasks)
    mlflow.log_param("alpha", alpha)

    dataset_len = len(dataset)

    distilation_loss = torch.zeros([1], dtype=torch.float, device=cfg["device"])
    gen_distilation_loss = distilation_loss.clone()
    new_distilation_loss = distilation_loss.clone()

    for epoch in range(cfg["max_epochs"]):
        train_loss = 0
        gen_distilation_losses = 0
        new_distilation_losses = 0
        distilation_losses = 0
        cross_entropy_losses = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            cfg["cl_optimizer"].zero_grad()

            if trained_tasks:
                with torch.no_grad():
                    gen_imgs, _ = cfg["generator"].generate(
                        batch_size=cfg["cl_batch_size"]
                    )
                    gen_output_old, _ = previous_model(gen_imgs.detach())
                    previous_output, _ = previous_model(imgs)

                gen_output, _ = cfg["classifier"](gen_imgs.detach())
                output, _ = cfg["classifier"](imgs)

                # loss for new data

                output_tasks = utils.filter_logits(cfg, output, task | trained_tasks)
                cross_entropy_loss = F.cross_entropy(output_tasks, labels)

                # distillation loss for generative old data

                gen_output_old = utils.filter_logits(cfg, gen_output_old, trained_tasks)
                gen_soft_target = F.softmax(gen_output_old / T, dim=1)

                gen_output = utils.filter_logits(cfg, gen_output, trained_tasks)
                gen_logp = F.log_softmax(gen_output / T, dim=1)

                gen_distilation_loss = -torch.mean(
                    torch.sum(gen_soft_target * gen_logp, dim=1)
                )

                # distillation loss for new data

                previous_output = utils.filter_logits(
                    cfg, previous_output, trained_tasks
                )
                soft_target = F.softmax(previous_output / T, dim=1)

                output = utils.filter_logits(cfg, output, trained_tasks)
                logp = F.log_softmax(output / T, dim=1)

                new_distilation_loss = -torch.mean(torch.sum(soft_target * logp, dim=1))

                distilation_loss = new_distilation_loss + gen_distilation_loss

                gen_distilation_losses += gen_distilation_loss.item()
                new_distilation_losses += new_distilation_loss.item()
            else:
                output, _ = cfg["classifier"](imgs)
                cross_entropy_loss = F.cross_entropy(output, labels)

            loss = (1 - alpha) * cross_entropy_loss + alpha * distilation_loss

            loss.backward()
            cfg["cl_optimizer"].step()
            train_loss += loss.item()
            cross_entropy_losses += cross_entropy_loss.item()
            distilation_losses += distilation_loss.item()

        train_loss /= dataset_len / cfg["cl_batch_size"]
        cross_entropy_losses /= dataset_len / cfg["cl_batch_size"]
        distilation_loss /= dataset_len / cfg["cl_batch_size"]
        gen_distilation_loss /= dataset_len / cfg["cl_batch_size"]
        new_distilation_loss /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_loss, step=epoch)
        mlflow.log_metric("Classification Loss", cross_entropy_losses, step=epoch)
        mlflow.log_metric("Distilation Loss", distilation_loss, step=epoch)
        mlflow.log_metric(
            "Gnenerator Distilation Loss", gen_distilation_loss, step=epoch
        )
        mlflow.log_metric("New Distilation Loss", new_distilation_loss, step=epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            model_checkpoint = {
                "model": cfg["classifier"].state_dict(),
                "trained_tasks": task | trained_tasks,
                "batch_size": cfg["cl_batch_size"],
            }
            torch.save(model_checkpoint, cfg["classifier_model_file"])
        else:
            patience_counter += 1

        mlflow.log_metric("Best loss", best_loss, step=epoch)
        mlflow.log_metric("Patience Counter", patience_counter, step=epoch)

        if patience_counter >= cfg["cl_max_patience"]:
            break

    del previous_model


def train_generator(cfg, task):

    replay_tasks = {}

    if cfg["classifier_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["classifier_model_file"])
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        cfg["classifier"].to(cfg["device"])

    if cfg["generator_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["generator_model_file"])
        cfg["generator"].load_state_dict(model_checkpoint["model"])
        replay_tasks = model_checkpoint["replay_tasks"]
        cfg["generator"].to(cfg["device"])

        task |= replay_tasks

        mlflow.log_param("generator replay task", task)

    mlflow.log_param("generator task", task)

    cfg["classifier"].eval()

    hooks = []
    for module in cfg["classifier"].modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(utils.DeepInversionFeatureHook(module))

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    dataset_len = len(task) * cfg["images_per_task"]
    gen_batches = int(dataset_len / cfg["gen_batch_size"])

    for epoch in range(cfg["max_epochs"]):
        cross_entropy_losses = 0
        activation_losses = 0
        information_entropy_losses = 0
        variation_regularization_losses = 0
        batchmorm_losses = 0
        g_losses = 0

        cfg["generator"].train()
        cfg["classifier"].eval()

        for _ in range(gen_batches):
            cfg["g_optimizer"].zero_grad()

            fakes, _ = cfg["generator"].generate(batch_size=cfg["gen_batch_size"])
            output_fake, features_fake = cfg["classifier"](fakes)

            output_fake = utils.filter_logits(cfg, output_fake, task)

            cross_entropy_loss = (
                F.cross_entropy(output_fake, torch.argmax(output_fake, dim=1))
                * CROSS_ENTROPY_LOSS_WEIGHT
            )

            activation_loss = -features_fake.abs().mean() * ACTIVATION_LOSS_WEIGHT

            softmax_o_T = F.softmax(output_fake, dim=1).mean(dim=0)
            information_entropy_loss = (
                softmax_o_T * torch.log10(softmax_o_T)
            ).sum() * INFORMATION_ENTROPY_LOSS_WEIGHT

            variation_regularization_loss = (
                ls.variation_regularization_loss(fakes)
                * VARIATION_REGULARIZATION_LOSS_WEIGHT
            )

            batchmorm_loss = (
                sum([hook.r_feature for hook in hooks if hook.r_feature is not None])
                / len(hooks)
            ) * BATCHMORM_LOSS_WEIGHT

            g_loss = (
                cross_entropy_loss
                + activation_loss
                + information_entropy_loss
                + variation_regularization_loss
                + batchmorm_loss
            )

            g_loss.backward()
            cfg["g_optimizer"].step()

            cross_entropy_losses += cross_entropy_loss.item()
            activation_losses += activation_loss.item()
            information_entropy_losses += information_entropy_loss.item()
            variation_regularization_losses += variation_regularization_loss.item()
            batchmorm_losses += batchmorm_loss.item()
            g_losses += g_loss.item()

        cross_entropy_losses /= dataset_len / cfg["gen_batch_size"]
        activation_losses /= dataset_len / cfg["gen_batch_size"]
        information_entropy_losses /= dataset_len / cfg["gen_batch_size"]
        variation_regularization_losses /= dataset_len / cfg["gen_batch_size"]
        batchmorm_losses /= dataset_len / cfg["gen_batch_size"]
        g_losses /= dataset_len / cfg["gen_batch_size"]

        mlflow.log_metric("classification loss", cross_entropy_losses, step=epoch)
        mlflow.log_metric("activation loss", activation_losses, step=epoch)
        mlflow.log_metric(
            "information entropy loss", information_entropy_losses, step=epoch
        )
        mlflow.log_metric(
            "variation regularization loss", variation_regularization_losses, step=epoch
        )
        mlflow.log_metric("batchmorm loss", batchmorm_losses, step=epoch)
        mlflow.log_metric("Total loss", g_losses, step=epoch)

        if g_losses < best_loss:
            best_loss = g_losses
            patience_counter = 0
            model_checkpoint = {
                "model": cfg["generator"].state_dict(),
                "replay_tasks": task,
                "batch_size": cfg["gen_batch_size"],
            }
            torch.save(model_checkpoint, cfg["generator_model_file"])
        else:
            patience_counter += 1

        mlflow.log_metric("Best loss", best_loss, step=epoch)
        mlflow.log_metric("Patience Counter", patience_counter, step=epoch)

        if patience_counter >= cfg["gen_max_patience"]:
            break

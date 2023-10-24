import math
from copy import deepcopy

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from dataset import TrainDatasetUnbalanced
from models import Classifier

import losses as ls

"""
Content adapted from
@article{smith2021always,
  author    = {Smith, James and Hsu, Yen-Chang and Balloch, Jonathan and Shen, Yilin and Jin, Hongxia and Kira, Zsolt},
  title     = {Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {9374-9384}
}
"""
# https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL/blob/main/experiments/cifar100-fivetask.sh#L38

MU = 1e-1
CROSS_ENTROPY_LOSS_WEIGHT = 1
CROSS_ENTROPY_LOSS_TEMP = 1e3
VARIANCE_PRIOR_WEIGHT = 1e-3
BATCHMORM_LOSS_WEIGHT = 5e1

torch.autograd.set_detect_anomaly(True)


def train_classifier(cfg, task):

    trained_tasks = {}
    previous_model = None
    previous_linear_layer = None

    if cfg["classifier_checkpoint_path"].is_file():

        model_checkpoint = torch.load(cfg["classifier_model_file"])

        cfg["classifier"].weights_init()
        cfg["classifier"].load_state_dict(model_checkpoint["model"])
        trained_tasks = model_checkpoint["trained_tasks"]
        cfg["classifier"].to(cfg["device"])

        previous_model, _ = utils.initialize_model_and_optimizer(cfg, Classifier)
        previous_model.load_state_dict(model_checkpoint["model"])
        previous_model.to(cfg["device"])
        previous_model.eval()

        previous_linear_layer = deepcopy(previous_model.fc)

    if cfg["generator_checkpoint_path"].is_file():
        model_checkpoint = torch.load(cfg["generator_model_file"])
        cfg["generator"].load_state_dict(model_checkpoint["model"])
        cfg["generator"].to(cfg["device"])
        cfg["generator"].eval()

    if trained_tasks:

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

    dataset_len = len(dataset)

    distilation_loss = torch.zeros([1], dtype=torch.float, device=cfg["device"])

    for epoch in range(cfg["max_epochs"]):
        train_loss = 0
        distilation_losses = 0
        cross_entropy_losses = 0
        cross_entropy_f1_losses = 0
        cross_entropy_f2_losses = 0
        cross_entropy_o_losses = 0
        distilation_o_losses = 0
        distilation_g_losses = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            utils.apply_prunning(cfg, task, trained_tasks)

            cfg["cl_optimizer"].zero_grad()

            output, features = cfg["classifier"](imgs)

            if trained_tasks:

                output_task = utils.filter_logits(cfg, output, task)

                cross_entropy_o_loss = F.cross_entropy(output_task, labels)

                with torch.no_grad():
                    gen_imgs, _ = cfg["generator"].generate(
                        batch_size=cfg["cl_batch_size"]
                    )
                    gen_output, gen_features = cfg["classifier"](gen_imgs.detach())
                    features_previous = previous_model(imgs, features_only=True)
                    gen_features_previous = previous_model(
                        gen_imgs.detach(), features_only=True
                    )

                gen_output = utils.filter_logits(cfg, gen_output, trained_tasks)
                gen_labels = torch.argmax(gen_output, dim=1)

                cross_entropy_f1_loss = F.cross_entropy(
                    cfg["classifier"].fc(features.detach()), labels
                )
                cross_entropy_f2_loss = F.cross_entropy(
                    cfg["classifier"].fc(gen_features.detach()), gen_labels
                )

                cross_entropy_loss = (
                    cross_entropy_o_loss + cross_entropy_f1_loss + cross_entropy_f2_loss
                )

                cross_entropy_o_losses += cross_entropy_o_loss.item()
                cross_entropy_f1_losses += cross_entropy_f1_loss.item()
                cross_entropy_f2_losses += cross_entropy_f2_loss.item()

                logits_kd = previous_linear_layer(features.detach())
                logits_kd_past = previous_linear_layer(features_previous.detach())

                distilation_o_loss = MU * ls.distillation_loss(
                    logits_kd, logits_kd_past, trained_tasks
                )

                logits_kd_gen = previous_linear_layer(gen_features)
                logits_kd_past_gen = previous_linear_layer(gen_features_previous)

                distilation_g_loss = MU * ls.distillation_loss(
                    logits_kd_gen, logits_kd_past_gen, trained_tasks
                )

                distilation_loss = distilation_o_loss + distilation_g_loss

                distilation_o_losses += distilation_o_loss.item()
                distilation_g_losses += distilation_g_loss.item()

            else:

                cross_entropy_o_loss = F.cross_entropy(output, labels)
                cross_entropy_loss = cross_entropy_o_loss
                cross_entropy_o_losses += cross_entropy_o_loss.item()

            loss = distilation_loss + cross_entropy_loss

            loss.backward()
            cfg["cl_optimizer"].step()
            train_loss += loss.item()
            cross_entropy_losses += cross_entropy_loss.item()
            distilation_losses += distilation_loss.item()

        train_loss /= dataset_len / cfg["cl_batch_size"]
        cross_entropy_losses /= dataset_len / cfg["cl_batch_size"]
        distilation_losses /= dataset_len / cfg["cl_batch_size"]
        cross_entropy_o_losses /= dataset_len / cfg["cl_batch_size"]
        cross_entropy_f1_losses /= dataset_len / cfg["cl_batch_size"]
        cross_entropy_f2_losses /= dataset_len / cfg["cl_batch_size"]
        distilation_o_losses /= dataset_len / cfg["cl_batch_size"]
        distilation_g_losses /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_loss, step=epoch)
        mlflow.log_metric("Classification Loss", cross_entropy_losses, step=epoch)
        mlflow.log_metric("Distilation Loss", distilation_losses, step=epoch)

        mlflow.log_metric("Classification O Loss", cross_entropy_o_losses, step=epoch)
        mlflow.log_metric("Classification F1 Loss", cross_entropy_f1_losses, step=epoch)
        mlflow.log_metric("Classification F2 Loss", cross_entropy_f2_losses, step=epoch)
        mlflow.log_metric("Distilation O Loss", distilation_o_losses, step=epoch)
        mlflow.log_metric("Distilation G Loss", distilation_g_losses, step=epoch)

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

    # print(model_checkpoint["trained_tasks"])
    del previous_model
    del previous_linear_layer


def train_generator(cfg, task):

    replay_tasks = {}

    mlflow.log_param("generator task", task)

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

    cfg["classifier"].eval()

    hooks = []
    for module in cfg["classifier"].modules():
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(utils.DeepInversionFeatureHook(module))

    smoothing = utils.Gaussiansmoothing(
        channels=cfg["img_channels"], kernel_size=cfg["smoothing_kernel_size"], sigma=5
    )

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    dataset_len = len(task) * cfg["images_per_task"]
    gen_batches = int(dataset_len / cfg["gen_batch_size"])

    for epoch in range(cfg["max_epochs"]):
        class_balance_losses = 0
        batchmorm_losses = 0
        cross_entropy_losses = 0
        smoothing_losses = 0
        g_losses = 0

        cfg["generator"].train()
        cfg["classifier"].eval()

        for _ in range(gen_batches):
            cfg["g_optimizer"].zero_grad()

            fakes, _ = cfg["generator"].generate(batch_size=cfg["gen_batch_size"])
            output_fake, _ = cfg["classifier"](fakes)

            output_fake = utils.filter_logits(cfg, output_fake, task)

            cross_entropy_loss = (
                F.cross_entropy(
                    output_fake / CROSS_ENTROPY_LOSS_TEMP,
                    torch.argmax(output_fake, dim=1),
                )
            ) * CROSS_ENTROPY_LOSS_WEIGHT

            softmax_o_T = F.softmax(output_fake, dim=1).mean(dim=0)
            softmax_o_T_log = F.log_softmax(output_fake, dim=1).mean(dim=0)
            class_balance_loss = (
                1.0 + (softmax_o_T * softmax_o_T_log / math.log(len(task))).sum()
            )

            batchmorm_loss = (
                sum([hook.r_feature for hook in hooks if hook.r_feature is not None])
                / len(hooks)
            ) * BATCHMORM_LOSS_WEIGHT

            fakes_smoothed = smoothing(F.pad(fakes, (1, 1, 1, 1), mode="reflect"))
            smoothing_loss = F.mse_loss(fakes, fakes_smoothed) * VARIANCE_PRIOR_WEIGHT

            g_loss = (
                cross_entropy_loss
                + class_balance_loss
                + batchmorm_loss
                + smoothing_loss
            )

            g_loss.backward()
            cfg["g_optimizer"].step()

            cross_entropy_losses += cross_entropy_loss.item()
            batchmorm_losses += batchmorm_loss.item()
            class_balance_losses += class_balance_loss.item()
            smoothing_losses += smoothing_loss.item()
            g_losses += g_loss.item()

        class_balance_losses /= dataset_len / cfg["gen_batch_size"]
        cross_entropy_losses /= dataset_len / cfg["gen_batch_size"]
        batchmorm_losses /= dataset_len / cfg["gen_batch_size"]
        smoothing_losses /= dataset_len / cfg["gen_batch_size"]
        g_losses /= dataset_len / cfg["gen_batch_size"]

        mlflow.log_metric("class balance loss", class_balance_losses, step=epoch)
        mlflow.log_metric("classification loss", cross_entropy_losses, step=epoch)
        mlflow.log_metric("batchmorm loss", batchmorm_losses, step=epoch)
        mlflow.log_metric("smoothing losses", smoothing_losses, step=epoch)
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

    # print(model_checkpoint["replay_tasks"])

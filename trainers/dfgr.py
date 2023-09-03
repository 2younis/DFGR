import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import utils
import losses as ls
from dataset import TrainDatasetUnbalanced


def train_classifier(cfg, task, adjust_replay):

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
        cfg["generator"].eval()

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    probabilities = None

    cfg["classifier"].train()

    if not replay_tasks:
        mix_ratio = 0
    else:
        mix_ratio = len(replay_tasks) / (len(replay_tasks) + len(task))

    dataset = TrainDatasetUnbalanced(cfg, task, dataset=cfg["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=cfg["cl_batch_size"], shuffle=True, drop_last=True
    )

    mlflow.log_param("classifier replay task", replay_tasks)
    mlflow.log_param("mix ratio", mix_ratio)
    mlflow.log_param("adjust replay probalilities", adjust_replay)

    dataset_len = len(dataset)

    for epoch in range(cfg["max_epochs"]):
        train_loss = 0
        replay_losses = 0
        real_losses = 0

        replay_epoch_losses = None
        replay_labels = None
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(cfg["device"]), labels.to(cfg["device"])

            cfg["cl_optimizer"].zero_grad()
            output, _ = cfg["classifier"](imgs)

            real_loss = ls.focal_loss(output, labels)

            if mix_ratio > 0:
                with torch.no_grad():
                    gen_imgs, gen_labels = cfg["generator"].generate(
                        replay_tasks,
                        cfg["cl_batch_size"],
                        trunc=cfg["truncation"],
                        probabilities=probabilities,
                    )
                gen_imgs, gen_labels = (
                    gen_imgs.to(cfg["device"]).detach(),
                    gen_labels.to(cfg["device"]).detach(),
                )

                gen_output, _ = cfg["classifier"](gen_imgs)
                replay_loss = F.cross_entropy(gen_output, gen_labels, reduction="none")

                if replay_labels is None:
                    replay_labels = gen_labels.detach().cpu()
                    replay_epoch_losses = replay_loss.detach().cpu()
                else:
                    replay_labels = torch.cat(
                        [replay_labels, gen_labels.detach().cpu()], dim=0
                    )
                    replay_epoch_losses = torch.cat(
                        [replay_epoch_losses, replay_loss.detach().cpu()], dim=0
                    )

                replay_loss = replay_loss.mean()

                loss = (1 - mix_ratio) * real_loss + mix_ratio * replay_loss
                replay_losses += replay_loss.item()

            else:
                loss = real_loss

            loss.backward()
            cfg["cl_optimizer"].step()
            train_loss += loss.item()
            real_losses += real_loss.item()

        if replay_labels is not None and adjust_replay:

            probabilities = utils.adjust_replay_probabilities(
                replay_epoch_losses, replay_labels, probabilities
            )

        train_loss /= dataset_len / cfg["cl_batch_size"]
        real_losses /= dataset_len / cfg["cl_batch_size"]
        replay_losses /= dataset_len / cfg["cl_batch_size"]

        mlflow.log_metric("Total Loss", train_loss, step=epoch)
        mlflow.log_metric("Real Loss", real_losses, step=epoch)
        mlflow.log_metric("Replay Loss", replay_losses, step=epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            model_checkpoint = {
                "model": cfg["classifier"].state_dict(),
                "trained_tasks": task | replay_tasks,
                "batch_size": cfg["cl_batch_size"],
            }
            torch.save(model_checkpoint, cfg["classifier_model_file"])
        else:
            patience_counter += 1

        mlflow.log_metric("Best loss", best_loss, step=epoch)
        mlflow.log_metric("Patience Counter", patience_counter, step=epoch)

        if patience_counter >= cfg["cl_max_patience"]:
            break

    utils.save_features(cfg)


def train_generator(cfg, task, generator_params):
    delta, alpha, beta, gamma, epsilon = generator_params

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
    mlflow.log_param("delta", delta)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("beta", beta)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("epsilon", epsilon)

    cfg["classifier"].eval()
    cfg["classifier"].register_hooks()

    smoothing = utils.Gaussiansmoothing(
        channels=cfg["img_channels"], kernel_size=cfg["smoothing_kernel_size"]
    )

    best_loss = float(cfg["max_loss"])
    patience_counter = 0

    if beta > 0:
        batchnorm_means, batchnorm_vars = None, None

        for module in cfg["classifier"].modules():
            if isinstance(module, nn.BatchNorm2d):

                if batchnorm_means is None:
                    batchnorm_means = module.running_mean
                    batchnorm_vars = module.running_var
                else:
                    batchnorm_means = torch.cat(
                        [batchnorm_means, module.running_mean], dim=0
                    )
                    batchnorm_vars = torch.cat(
                        [batchnorm_vars, module.running_var], dim=0
                    )

        batchnorm_means = batchnorm_means[cfg["img_channels"] :]
        batchnorm_vars = batchnorm_vars[cfg["img_channels"] :]

    if alpha > 0:
        features_dict = torch.load(cfg["features_file"])

    dataset_len = len(task) * cfg["images_per_task"]
    gen_batches = int(dataset_len / cfg["gen_batch_size"])

    class_loss = cfg["class_loss"]
    features_loss = cfg["features_loss"]
    batchmorm_loss = cfg["batchmorm_loss"]
    div_loss = cfg["div_loss"]
    smoothing_loss = cfg["smoothing_loss"]

    for epoch in range(cfg["max_epochs"]):
        div_losses = 0
        features_losses = 0
        batchmorm_losses = 0
        class_losses = 0
        smoothing_losses = 0
        g_losses = 0

        cfg["generator"].train()
        cfg["classifier"].eval()

        for _ in range(gen_batches):
            cfg["g_optimizer"].zero_grad()

            fakes, labels = cfg["generator"].generate(task, cfg["gen_batch_size"])
            fakes, labels = fakes.to(cfg["device"]), labels.to(cfg["device"])

            output_fake, fake_h = cfg["classifier"](fakes)

            if delta > 0:
                class_loss = F.cross_entropy(output_fake, labels)

            if epsilon > 0:
                fakes_smoothed = smoothing(F.pad(fakes, (1, 1, 1, 1), mode="reflect"))
                smoothing_loss = F.mse_loss(fakes, fakes_smoothed)

            if alpha > 0:
                sigma_1, mu1 = torch.var_mean(fake_h, dim=0, unbiased=False)
                sigma_2, mu2 = ls.merge_gaussians(features_dict, labels)

                features_loss = torch.norm(mu1 - mu2.to(cfg["device"])) + torch.norm(
                    sigma_1 - sigma_2.to(cfg["device"])
                )

            if beta > 0:
                batch_means, batch_vars = None, None

                for hook in cfg["classifier"].hooks:
                    if hook.mean is not None:
                        if batch_means is None:
                            batch_means = hook.mean
                            batch_vars = hook.var
                        else:
                            batch_means = torch.cat([batch_means, hook.mean], dim=0)
                            batch_vars = torch.cat([batch_vars, hook.var], dim=0)

                batchmorm_loss = torch.norm(batch_means - batchnorm_means) + torch.norm(
                    batch_vars - batchnorm_vars
                )

            if gamma > 0:
                div_loss = -ls.js_divergence(fakes, cfg["num_div_samples"])

            g_loss = (
                (delta * class_loss)
                + (alpha * features_loss)
                + (beta * batchmorm_loss)
                + (gamma * div_loss)
                + (epsilon * smoothing_loss)
            )

            g_loss.backward()
            cfg["g_optimizer"].step()

            class_losses += class_loss.item()
            features_losses += features_loss.item()
            batchmorm_losses += batchmorm_loss.item()
            div_losses += div_loss.item()
            smoothing_losses += smoothing_loss.item()
            g_losses += g_loss.item()

        div_losses /= dataset_len / cfg["gen_batch_size"]
        class_losses /= dataset_len / cfg["gen_batch_size"]
        features_losses /= dataset_len / cfg["gen_batch_size"]
        batchmorm_losses /= dataset_len / cfg["gen_batch_size"]
        smoothing_losses /= dataset_len / cfg["gen_batch_size"]
        g_losses /= dataset_len / cfg["gen_batch_size"]

        mlflow.log_metric("divergence loss", div_losses, step=epoch)
        mlflow.log_metric("classification loss", class_losses, step=epoch)
        mlflow.log_metric("features loss", features_losses, step=epoch)
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

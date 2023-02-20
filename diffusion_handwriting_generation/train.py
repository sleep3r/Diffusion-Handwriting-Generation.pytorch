import logging
import time

import torch
import torch.nn as nn

from diffusion_handwriting_generation.config import (
    DLConfig,
    config_entrypoint,
    object_from_dict,
)
from diffusion_handwriting_generation.dataset import IAMDataset
from diffusion_handwriting_generation.loss import loss_fn
from diffusion_handwriting_generation.model import DiffusionModel
from diffusion_handwriting_generation.utils.experiment import log_artifacts, prepare_exp
from diffusion_handwriting_generation.utils.nn import get_alphas, get_beta_set


def train_step(x, pen_lifts, text, style_vectors, glob_args):
    model, alpha_set, bce, train_loss, optimizer = glob_args
    alphas = get_alphas(len(x), alpha_set)
    eps = torch.randn_like(x)
    x_perturbed = torch.sqrt(alphas) * x
    x_perturbed += torch.sqrt(1 - alphas) * eps

    optimizer.zero_grad()
    score, pl_pred, att = model(
        x_perturbed, text, torch.sqrt(alphas), style_vectors, training=True
    )
    loss = loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    return score, att


def train(cfg: DLConfig, meta: dict, logger: logging.Logger) -> None:
    model = DiffusionModel(
        num_layers=cfg.training_args.num_attlayers,
        c1=cfg.training_args.channels,
        c2=cfg.training_args.channels * 3 // 2,
        c3=cfg.training_args.channels * 2,
        drop_rate=cfg.training_args.dropout,
    )
    optimizer = object_from_dict(cfg.optimizer, params=model.parameters())

    train_dataset = IAMDataset(
        data_dir=cfg.experiment.data_dir,
        kind="train",
        splits_file=cfg.experiment.splits_file,
        **cfg.experiment.dataset_args,
    )
    valid_dataset = IAMDataset(
        data_dir=cfg.experiment.data_dir,
        kind="validation",
        splits_file=cfg.experiment.splits_file,
        **cfg.experiment.dataset_args,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training_args.batch_size,
        shuffle=True,
        num_workers=cfg.training_args.num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.training_args.batch_size,
        shuffle=False,
        num_workers=cfg.training_args.num_workers,
        pin_memory=True,
    )

    s = time.time()
    bce = nn.BCELoss()
    train_loss = []
    beta_set = get_beta_set()
    alpha_set = torch.cumprod(1 - beta_set)

    try:
        logger.info(
            f'Starting train tagger, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n'
        )
        for count, batch in enumerate(train_loader):
            strokes, text, style_vectors = (
                batch["strokes"],
                batch["text"],
                batch["style"],
            )
            strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
            glob_args = model, alpha_set, bce, train_loss, optimizer

            train_step(strokes, pen_lifts, text, style_vectors, glob_args)

            if (count + 1) % cfg.training_args.print_every == 0:
                logger.info(
                    "Iteration %d, Loss %f, Time %ds"
                    % (count + 1, sum(train_loss) / len(train_loss), time.time() - s)
                )
                train_loss = []

            if (count + 1) % cfg.training_args.save_every == 0:
                save_path = "./weights/model_step%d.pth" % (count + 1)
                torch.save(model.state_dict(), save_path)

            if count >= cfg.training_args.steps:
                torch.save(model.state_dict(), "./weights/model.pth")
                break
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")


def main(cfg: DLConfig) -> None:
    meta, logger, training = prepare_exp(cfg)

    logger.info(f"Config:\n{cfg.pretty_text}\n")

    train(cfg, meta, logger)

    log_artifacts(cfg, meta)


if __name__ == "__main__":
    config: DLConfig = config_entrypoint()
    main(config)

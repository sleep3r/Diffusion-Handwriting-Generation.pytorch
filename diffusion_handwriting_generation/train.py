import logging
import time

import torch

from diffusion_handwriting_generation.checkpoint import save_checkpoint
from diffusion_handwriting_generation.config import (
    DLConfig,
    config_entrypoint,
    object_from_dict,
)
from diffusion_handwriting_generation.dataset import IAMDataset
from diffusion_handwriting_generation.loss import loss_fn
from diffusion_handwriting_generation.model import DiffusionModel
from diffusion_handwriting_generation.scheduler import InvSqrtScheduledOptim
from diffusion_handwriting_generation.utils.clip_grad import dispatch_clip_grad
from diffusion_handwriting_generation.utils.experiment import log_artifacts, prepare_exp
from diffusion_handwriting_generation.utils.nn import get_alphas, get_beta_set


def train_step(
    cfg: DLConfig,
    device: torch.device,
    x: torch.Tensor,
    pen_lifts: torch.Tensor,
    text: torch.Tensor,
    style_vectors: torch.Tensor,
    glob_args: tuple[DiffusionModel, torch.Tensor, list[float], InvSqrtScheduledOptim],
) -> None:
    model, alpha_set, train_loss, optimizer = glob_args

    alphas = get_alphas(len(x), alpha_set)
    eps = torch.randn_like(x)

    x_perturbed = (
        torch.sqrt(alphas).unsqueeze(-1) * x
        + torch.sqrt(1 - alphas).unsqueeze(-1) * eps
    )

    pen_lifts = pen_lifts.to(device)
    text = text.to(device)
    style_vectors = style_vectors.to(device)
    x_perturbed = x_perturbed.to(device)
    alphas = alphas.to(device)
    eps = eps.to(device)

    strokes_pred, pen_lifts_pred, att = model(
        x_perturbed,
        text,
        torch.sqrt(alphas),
        style_vectors,
    )

    loss = loss_fn(eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas)

    optimizer.zero_grad()
    loss.backward()

    if cfg.training_args.clip_grad is not None:
        dispatch_clip_grad(
            model.parameters(),
            value=cfg.training_args.clip_grad,
        )

    optimizer.step_and_update_lr()

    train_loss.append(loss.item())


def train(cfg: DLConfig, meta: dict, logger: logging.Logger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionModel(
        num_layers=cfg.training_args.att_layers_num,
        c1=cfg.training_args.channels,
        c2=cfg.training_args.channels * 3 // 2,
        c3=cfg.training_args.channels * 2,
        drop_rate=cfg.training_args.dropout,
    )
    model.to(device)
    model.train()

    optimizer = InvSqrtScheduledOptim(
        optimizer=object_from_dict(cfg.optimizer, params=model.parameters()),
        lr_mul=1.0,
        d_model=cfg.training_args.channels,
        n_warmup_steps=cfg.training_args.warmup_steps,
    )

    logger.info("Loading data...")
    train_dataset = IAMDataset(
        data_dir=cfg.experiment.data_dir,
        kind="train",
        splits_file=cfg.experiment.splits_file,
        max_files=cfg.training_args.max_files,
        **cfg.dataset_args,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training_args.batch_size,
        num_workers=cfg.training_args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    s = time.time()
    train_loss: list[float] = []
    beta_set = get_beta_set()
    alpha_set = torch.cumprod(1 - beta_set, dim=0)

    logger.info(
        f'Starting train model, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n',
    )
    try:
        count = 0
        while True:
            batch = next(iter(train_loader))
            count += 1

            strokes, text, style_vectors = (
                batch["strokes"],
                batch["text"],
                batch["style"],
            )
            strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

            glob_args = model, alpha_set, train_loss, optimizer
            train_step(cfg, device, strokes, pen_lifts, text, style_vectors, glob_args)

            if (count + 1) % cfg.training_args.log_freq == 0:
                logger.info(
                    f"Step {count + 1} | "
                    f"Loss: {sum(train_loss) / len(train_loss):.3f} | "
                    f"Time: {time.time() - s:.3f} sec",
                )
                train_loss = []

            if (count + 1) % cfg.training_args.save_freq == 0:
                checkpoint_path = meta["exp_dir"] / f"checkpoint_{count + 1}.pth"
                logger.info("Saving checkpoint...")
                save_checkpoint(model, checkpoint_path)

            if count >= cfg.training_args.steps:
                logger.info("Training finished, saving model weights.")
                model_path = meta["exp_dir"] / "model_final.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(str(model_path))
                break
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        save_checkpoint(model, meta["exp_dir"] / "checkpoint_last.pth")
        torch.save(model.state_dict(), meta["exp_dir"] / "model_last.pth")


def main(cfg: DLConfig) -> None:
    meta, logger = prepare_exp(cfg)

    logger.info(f"Config:\n{cfg.pretty_text}\n")

    train(cfg, meta, logger)

    log_artifacts(cfg, meta)


if __name__ == "__main__":
    config: DLConfig = config_entrypoint()
    main(config)

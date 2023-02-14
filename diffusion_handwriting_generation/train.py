import logging
import time

import torch
import torch.nn as nn

from diffusion_handwriting_generation.config import DLConfig, load_config, object_from_dict
from diffusion_handwriting_generation.dataset import preprocess_data
from diffusion_handwriting_generation.loss import loss_fn
from diffusion_handwriting_generation.model import DiffusionWriter
from diffusion_handwriting_generation.preprocessing import create_dataset
from diffusion_handwriting_generation.text_style import StyleExtractor
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
    model = DiffusionWriter(
        num_layers=cfg.training_args.num_attlayers,
        c1=cfg.training_args.channels,
        c2=cfg.training_args.channels * 3 // 2,
        c3=cfg.training_args.channels * 2,
        drop_rate=cfg.training_args.dropout
    )
    optimizer = object_from_dict(cfg.optimizer, params=model.parameters())

    strokes, texts, samples = preprocess_data(
        path="./data/train_strokes.p",
        max_text_len=cfg.training_args.textlen,
        max_seq_len=cfg.training_args.seqlen,
        img_width=cfg.training_args.width,
        img_height=96
    )

    style_extractor = StyleExtractor()
    loader = create_dataset(
        strokes, texts, samples, style_extractor, cfg.training_args.batchsize, 3000
    )

    s = time.time()
    bce = nn.BCELoss()
    train_loss = []
    beta_set = get_beta_set()
    alpha_set = torch.cumprod(1 - beta_set)

    logger.info(
        f'Starting train tagger, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n'
    )
    for count, (strokes, text, style_vectors) in enumerate(loader.repeat(5000)):
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


def main(cfg: DLConfig) -> None:
    meta, logger, training = prepare_exp(cfg)

    logger.info(f"Config:\n{cfg.pretty_text}\n")

    try:
        train(cfg, meta, logger)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    log_artifacts(cfg, meta)


if __name__ == "__main__":
    config: DLConfig = load_config()
    main(config)

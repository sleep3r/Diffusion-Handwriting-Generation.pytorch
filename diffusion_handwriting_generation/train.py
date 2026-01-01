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
from diffusion_handwriting_generation.utils.nn import get_alphas, get_beta_set, get_device


class TrainingLoop:
    def __init__(self, cfg: DLConfig):
        self.cfg = cfg
        self.device = get_device()

    def train_step(
        self,
        batch: torch.Tensor,
        model: DiffusionModel,
        alpha_set: torch.Tensor,
        train_loss: list,
        score_loss_list: list,
        pen_lifts_loss_list: list,
        optimizer: InvSqrtScheduledOptim,
    ):
        x, pen_lifts, text, style_vectors = self.process_batch(batch)

        alphas = get_alphas(len(x), alpha_set).to(self.device)
        eps = torch.randn_like(x)

        x_perturbed = (
            torch.sqrt(alphas).unsqueeze(-1) * x + torch.sqrt(1 - alphas).unsqueeze(-1) * eps
        )

        optimizer.zero_grad()
        strokes_pred, pen_lifts_pred, _ = model(
            x_perturbed,
            text,
            torch.sqrt(alphas),
            style_vectors,
        )
        loss, score_loss, pen_lifts_loss = loss_fn(
            eps, strokes_pred, pen_lifts, pen_lifts_pred, alphas
        )
        loss.backward()

        if self.cfg.training_args.clip_grad is not None:
            dispatch_clip_grad(
                model.parameters(),
                value=self.cfg.training_args.clip_grad,
            )

        optimizer.step_and_update_lr()

        train_loss.append(loss.item())
        score_loss_list.append(score_loss.item())
        pen_lifts_loss_list.append(pen_lifts_loss.item())

    def process_batch(self, batch):
        strokes, text, style_vectors = (
            batch["strokes"],
            batch["text"],
            batch["style"],
        )
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2]

        strokes = strokes.to(self.device)
        pen_lifts = pen_lifts.to(self.device)
        text = text.to(self.device)
        style_vectors = style_vectors.to(self.device)
        return strokes, pen_lifts, text, style_vectors

    def train(self, meta: dict, logger: logging.Logger):
        model, optimizer, train_loader = self.prepare_training()
        s = time.time()
        train_loss: list[float] = []
        score_loss_list: list[float] = []
        pen_lifts_loss_list: list[float] = []
        beta_set = get_beta_set()
        alpha_set = torch.cumprod(1 - beta_set, dim=0)

        logger.info(
            f"Starting train model, host: {meta['host_name']}, exp_dir: {meta['exp_dir']}\n",
        )
        try:
            count = 0
            while True:
                batch = next(iter(train_loader))
                count += 1

                self.train_step(
                    batch,
                    model,
                    alpha_set,
                    train_loss,
                    score_loss_list,
                    pen_lifts_loss_list,
                    optimizer,
                )

                if (count + 1) % self.cfg.training_args.log_freq == 0:
                    logger.info(
                        f"Step {count + 1} | "
                        f"Loss: {sum(train_loss) / len(train_loss):.3f} | "
                        f"Score: {sum(score_loss_list) / len(score_loss_list):.3f} | "
                        f"Pen: {sum(pen_lifts_loss_list) / len(pen_lifts_loss_list):.3f} | "
                        f"Time: {time.time() - s:.3f} sec",
                    )
                    train_loss = []
                    score_loss_list = []
                    pen_lifts_loss_list = []

                if (count + 1) % self.cfg.training_args.save_freq == 0:
                    checkpoint_path = meta["exp_dir"] / f"checkpoint_{count + 1}.pth"
                    logger.info("Saving checkpoint...")
                    save_checkpoint(model, checkpoint_path)

                if count >= self.cfg.training_args.steps:
                    logger.info("Training finished, saving model weights.")
                    model_path = meta["exp_dir"] / "model_final.pth"
                    torch.save(model.state_dict(), model_path)
                    logger.info(str(model_path))
                    break
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            save_checkpoint(model, meta["exp_dir"] / "checkpoint_last.pth")
            torch.save(model.state_dict(), meta["exp_dir"] / "model_last.pth")

    def prepare_training(self):
        model = DiffusionModel(
            num_layers=self.cfg.training_args.att_layers_num,
            c1=self.cfg.training_args.channels,
            c2=self.cfg.training_args.channels * 3 // 2,
            c3=self.cfg.training_args.channels * 2,
            drop_rate=self.cfg.training_args.dropout,
        )
        model.to(self.device)
        model.train()

        optimizer = InvSqrtScheduledOptim(
            optimizer=object_from_dict(self.cfg.optimizer, params=model.parameters()),
            lr_mul=1.0,
            d_model=self.cfg.training_args.channels * 2,
            n_warmup_steps=self.cfg.training_args.warmup_steps,
        )

        train_dataset = IAMDataset(
            data_dir=self.cfg.experiment.data_dir,
            kind="train",
            splits_file=self.cfg.experiment.splits_file,
            max_files=self.cfg.training_args.max_files,
            **self.cfg.dataset_args,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.training_args.batch_size,
            num_workers=self.cfg.training_args.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return model, optimizer, train_loader


def main(cfg: DLConfig) -> None:
    train_loop = TrainingLoop(cfg)

    meta, logger = prepare_exp(cfg)

    logger.info(f"Config:\n{cfg.pretty_text}\n")

    train_loop.train(meta, logger)

    log_artifacts(cfg, meta)


if __name__ == "__main__":
    config: DLConfig = config_entrypoint()
    main(config)

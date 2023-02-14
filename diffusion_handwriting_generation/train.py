import time

import fire
import torch
import torch.nn as nn

from diffusion_handwriting_generation.dataset import preprocess_data
from diffusion_handwriting_generation.loss import loss_fn
from diffusion_handwriting_generation.model import DiffusionWriter
from diffusion_handwriting_generation.preprocessing import create_dataset
from diffusion_handwriting_generation.text_style import StyleExtractor
from diffusion_handwriting_generation.tokenizer import Tokenizer
from diffusion_handwriting_generation.utils.helpers import get_alphas, get_beta_set


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


def train(
    dataset, iterations, model, optimizer, alpha_set, print_every=1000, save_every=10000
):
    s = time.time()
    bce = nn.BCELoss()
    train_loss = []
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        glob_args = model, alpha_set, bce, train_loss, optimizer
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors, glob_args)

        if (count + 1) % print_every == 0:
            print(
                "Iteration %d, Loss %f, Time %ds"
                % (count + 1, sum(train_loss) / len(train_loss), time.time() - s)
            )
            train_loss = []

        if (count + 1) % save_every == 0:
            save_path = "./weights/model_step%d.pth" % (count + 1)
            torch.save(model.state_dict(), save_path)

        if count >= iterations:
            torch.save(model.state_dict(), "./weights/model.pth")
            break


def main(
    steps: int = 60000,
    batchsize: int = 96,
    seqlen: int = 480,
    textlen: int = 50,
    width: int = 1400,
    warmup: int = 10000,
    dropout: float = 0.0,
    num_attlayers: int = 2,
    channels: int = 128,
    print_every: int = 1000,
    save_every: int = 10000,
) -> None:
    """
    Trains a model with given hyperparameters.

    Args:
        steps (int): number of training steps. Defaults to 60000;
        batchsize (int): batch size. Defaults to 96;
        seqlen (int): sequence length during training. Defaults to 480;
        textlen (int): text length during training. Defaults to 50;
        width (int): offline image width. Defaults to 1400;
        warmup (int): number of warmup steps. Defaults to 10000;
        dropout (float): dropout rate. Defaults to 0.0;
        num_attlayers (int): number of attentional layers at lowest resolution. Defaults to 2;
        channels (int): number of channels in first layer. Defaults to 128;
        print_every (int): show train loss every n iters. Defaults to 1000;
        save_every (int): save checkpoint every n iters. Defaults to 10000.
    """
    NUM_STEPS = steps
    BATCH_SIZE = batchsize
    MAX_SEQ_LEN = seqlen
    MAX_TEXT_LEN = textlen
    WIDTH = width
    DROP_RATE = dropout
    NUM_ATTLAYERS = num_attlayers
    WARMUP_STEPS = warmup
    PRINT_EVERY = print_every
    SAVE_EVERY = save_every
    C1 = channels
    C2 = C1 * 3 // 2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN % 8) + 8

    BUFFER_SIZE = 3000
    L = 60
    tokenizer = Tokenizer()
    beta_set = get_beta_set()
    alpha_set = torch.cumprod(1 - beta_set)

    style_extractor = StyleExtractor()
    model = DiffusionWriter(
        num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.003
    )  # beta_1=0.9, beta_2=0.98, clipnorm=100

    path = "./data/train_strokes.p"
    strokes, texts, samples = preprocess_data(
        path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, 96
    )
    dataset = create_dataset(
        strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE
    )

    train(dataset, NUM_STEPS, model, optimizer, alpha_set, PRINT_EVERY, SAVE_EVERY)


if __name__ == "__main__":
    fire.Fire(main)

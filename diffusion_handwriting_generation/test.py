import os
from typing import List

import fire
from matplotlib import pyplot as plt
import numpy as np
import torch

from diffusion_handwriting_generation.model import DiffusionWriter
from diffusion_handwriting_generation.preprocessing import read_img
from diffusion_handwriting_generation.text_style import StyleExtractor
from diffusion_handwriting_generation.tokenizer import Tokenizer
from diffusion_handwriting_generation.utils.helpers import (
    get_beta_set,
    new_diffusion_step,
    show,
    standard_diffusion_step,
)


def run_batch_inference(
    model: torch.nn.Module,
    beta_set: List[float],
    text: str | List[str],
    style: torch.Tensor,
    tokenizer: Tokenizer,
    time_steps: int = 480,
    diffusion_mode: str = "new",
    show_every: List[int] | None = None,
    show_samples: bool = True,
    path: str | None = None,
) -> np.ndarray:
    """
    Runs batch inference on the input data with the given model.


        model (torch.nn.Module): model to run inference with;
        beta_set (List[float]): list of diffusion coefficients;
        text (str, List[str]): input text(s) to generate;
        style (torch.Tensor): tensor of the style to be applied;
        tokenizer (Tokenizer): tokenizer to use. Defaults to None;
        time_steps (int, optional): number of time steps for the diffusion process. Defaults to 480;
        diffusion_mode (str, optional): diffusion mode (standard or new). Defaults to "new";
        show_every (List[int], optional): show a plot of attention matrix every i-th beta. Defaults to None;
        show_samples (bool, optional): show generated samples. Defaults to True;
        path (str, optional): path to save the generated samples. Defaults to None.

    Returns:
        np.ndarray: generated samples in numpy format.
    """
    if isinstance(text, str):
        text = torch.tensor([tokenizer.encode(text) + [1]])
    elif isinstance(text, list) and isinstance(text[0], str):
        tmp = [tokenizer.encode(i) + [1] for i in text]
        text = torch.tensor(tmp)

    bs = text.shape[0]
    L = len(beta_set)
    alpha_set = (1 - torch.tensor(beta_set)).prod()
    x = torch.randn((bs, time_steps, 2))

    for i in range(L - 1, -1, -1):
        alpha = alpha_set[i] * torch.ones((bs, 1, 1))
        beta = beta_set[i] * torch.ones((bs, 1, 1))
        a_next = alpha_set[i - 1] if i > 0 else 1.0
        model_out, pen_lifts, att = model(x, text, torch.sqrt(alpha), style)
        if diffusion_mode == "standard":
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i))
        else:
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)

        if show_every is not None and i in show_every:
            plt.imshow(att[0][0].detach().numpy())
            plt.show()

    x = torch.cat([x, pen_lifts], dim=-1)
    for i in range(bs):
        show(x[i], scale=1, show_output=show_samples, name=path)
    return x.detach().numpy()


def main(
    textstring: str,
    writersource: str | None = None,
    name: str | None = None,
    diffmode: str = "new",
    show: bool = False,
    weights: str = "./weights/model_weights.h5",
    seqlen: int | None = None,
    num_attlayers: int = 2,
    channels: int = 128,
) -> None:
    """
    Generates text based on a given style.

    Parameters:
        textstring (str): text you want to generate;
        writersource (str, optional): path of the image of the desired writer. Will use a random image from ./assets if not specified. (default: None);
        name (str, optional): path for the generated image. The image will not be saved if not specified. (default: None);
        diffmode (str, optional): kind of y_t-1 prediction to use. Use 'standard' for Eq 9 in the paper, it will default to the prediction in Eq 12. (default: 'new');
        show (bool, optional): whether to show the sample in a popup from matplotlib. (default: False);
        weights (str, optional): the path of the loaded weights. (default: './weights/model_weights.h5');
        seqlen (int, optional): number of timesteps in the generated sequence. Default is 16 * the length of the text. (default: None);
        num_attlayers (int, optional): number of attentional layers at the lowest resolution. Only change this if the loaded model was trained with that hyperparameter. (default: 2);
        channels (int, optional): number of channels at the lowest resolution. Only change this if the loaded model was trained with that hyperparameter. (default: 128).

    Returns:
        None
    """
    timesteps = len(textstring) * 16 if seqlen is None else seqlen
    timesteps = timesteps - (timesteps % 8) + 8
    # must be divisible by 8 due to downsampling layers

    if writersource is None:
        assetdir = os.listdir("./assets")
        sourcename = "./assets/" + assetdir[np.random.randint(0, len(assetdir))]
    else:
        sourcename = writersource

    L = 60
    tokenizer = Tokenizer()
    beta_set = get_beta_set()
    alpha_set = 1 - beta_set.prod()

    C1 = channels
    C2 = C1 * 3 // 2
    C3 = C1 * 2
    style_extractor = StyleExtractor()
    model = DiffusionWriter(num_layers=num_attlayers, c1=C1, c2=C2, c3=C3)

    _stroke = torch.randn(1, 400, 2)
    _text = torch.randint(50, size=(1, 40), dtype=torch.int32)
    _noise = torch.rand(1, 1)
    _style_vector = torch.randn(1, 14, 1280)
    _ = model(_stroke, _text, _noise, _style_vector)
    # we have to call the model on input first
    model.load_state_dict(torch.load(weights))

    writer_img = read_img(sourcename, 96).unsqueeze(0)
    style_vector = style_extractor(writer_img)
    run_batch_inference(
        model,
        beta_set,
        textstring,
        style_vector,
        tokenizer=tokenizer,
        time_steps=timesteps,
        diffusion_mode=diffmode,
        show_samples=show,
        path=name,
    )


if __name__ == "__main__":
    fire.Fire(main)

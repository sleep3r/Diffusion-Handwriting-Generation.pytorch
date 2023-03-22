import fire
import torch
from tqdm import tqdm

from diffusion_handwriting_generation.checkpoint import load_model
from diffusion_handwriting_generation.text_style import StyleExtractor
from diffusion_handwriting_generation.tokenizer import Tokenizer
from diffusion_handwriting_generation.utils.io import read_img
from diffusion_handwriting_generation.utils.nn import (
    get_beta_set,
    new_diffusion_step,
    standard_diffusion_step,
)
from diffusion_handwriting_generation.utils.vis import show_strokes


def infer(
    prompt: str,
    source: str,
    config_path: str,
    checkpoint_path: str,
    diffusion_mode: str = "standart",
):
    model, device = load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    tokenizer = Tokenizer()
    beta_set = get_beta_set()
    style_extractor = StyleExtractor()

    writer_img = read_img(source, 96)[None, None, :]
    style_vector = style_extractor(writer_img)[None, ...]

    time_steps = len(prompt) * 16
    time_steps = time_steps - (time_steps % 8) + 8

    text = torch.tensor([tokenizer.encode(prompt) + [1]])

    bs = text.shape[0]
    alpha_set = torch.cumprod(1 - beta_set, dim=0)
    x = torch.randn((bs, time_steps, 2))

    for i in tqdm(range(len(beta_set) - 1, -1, -1)):
        alpha = alpha_set[i] * torch.ones((bs, 1, 1))
        beta = beta_set[i] * torch.ones((bs, 1, 1))
        a_next = alpha_set[i - 1] if i > 0 else torch.Tensor(1.0)

        model_out, pen_lifts, att = model(x, text, torch.sqrt(alpha), style_vector)

        if diffusion_mode == "standard":
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i))
        else:
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)

    x = torch.cat((x, pen_lifts.unsqueeze(2)), dim=2)

    show_strokes(x[0].detach().numpy(), scale=1)


if __name__ == "__main__":
    fire.Fire(infer)

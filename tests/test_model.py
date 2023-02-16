import torch

from diffusion_handwriting_generation import DiffusionModel


def test_forward_pass():
    nlayers = 2
    C1 = 128
    C2 = C1 * 3 // 2
    C3 = C1 * 2

    model = DiffusionModel(nlayers, C1, C2, C3)

    strokes = torch.rand(1, 400, 2)
    text = (torch.rand(1, 40) < 0.25).int()
    sigma = torch.rand(1, 1)
    style_vector = torch.rand(1, 14, 1280)

    model(strokes, text, sigma, style_vector)

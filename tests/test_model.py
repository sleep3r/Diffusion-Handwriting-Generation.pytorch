import torch

from diffusion_handwriting_generation import DiffusionModel


def test_forward_pass():
    nlayers = 2
    C1 = 128
    C2 = C1 * 3 // 2
    C3 = C1 * 2

    model = DiffusionModel(nlayers, C1, C2, C3)

    BS = 8

    strokes = torch.rand(BS, 400, 2)
    text = (torch.rand(BS, 40) < 0.25).int()
    sigma = torch.rand(BS, 1)
    style_vector = torch.rand(BS, 1, 1280)

    model(strokes, text, sigma, style_vector)

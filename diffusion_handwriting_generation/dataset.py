import torch
from torch.utils.data import Dataset


def create_dataloader(
    strokes: torch.Tensor,
    texts: torch.Tensor,
    samples: torch.Tensor,
    style_extractor: callable,
    batch_size: int,
    buffer_size: int,
) -> torch.utils.data.DataLoader:
    """
    Creates a PyTorch dataloader from the given inputs.

    Args:
        strokes (torch.Tensor): strokes data;
        texts (torch.Tensor): texts data;
        samples (torch.Tensor): samples data;
        style_extractor (callable): function that extracts the style vector from the samples;
        batch_size (int): batch size to use when creating the dataset;
        buffer_size (int): buffer size to use when shuffling the dataset.

    Returns:
        torch.utils.data.Dataset: created PyTorch dataset.
    """
    # we DO NOT SHUFFLE here, because we will shuffle later
    samples = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    style_vectors = style_extractor(samples)
    style_vectors = style_vectors.reshape(-1, style_vectors.shape[2])

    dataset = torch.utils.data.TensorDataset(strokes, texts, style_vectors)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=buffer_size,
        drop_last=True,
    )
    return loader

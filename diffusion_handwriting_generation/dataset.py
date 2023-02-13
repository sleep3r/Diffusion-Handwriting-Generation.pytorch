import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_stroke_seq(x: np.ndarray, maxlength: int) -> np.ndarray | None:
    """
    Pads the input stroke sequence `x` to `maxlength` by adding zeros
    for the coordinate values and ones for the pen states.
    If the length of `x` is greater than `maxlength` or if the maximum
    absolute value of the coordinate values is greater than 15, returns None.

    Args:
        x (np.ndarray): input stroke sequence, shape (N, 2);
        maxlength (int): desired length of the padded sequence.

    Returns:
        (np.ndarray, optional): padded sequence, shape (`maxlength`, 3) or None if the length
                                of `x` is greater than `maxlength` or if the maximum absolute
                                value of the coordinate values is greater than 15.
    """
    if len(x) > maxlength or np.amax(np.abs(x)) > 15:
        return None

    zeros = np.zeros((maxlength - len(x), 2))
    ones = np.ones((maxlength - len(x), 1))
    padding = np.concatenate((zeros, ones), axis=-1)
    x = np.concatenate((x, padding)).astype("float32")
    return x


def pad_img(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Pads the input image `img` to `width` by adding white pixels
    to the right of the image.

    Args:
        img (np.ndarray): input image, shape (`height`, `W`, 1);
        width (int): desired width of the padded image;
        height (int): desired height of the padded image.

    Returns:
        np.ndarray: padded image, shape (`height`, `width`, 1).
    """
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img


def preprocess_data(
    path: str,
    max_text_len: int,
    max_seq_len: int,
    img_width: int,
    img_height: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Preprocess the data by loading it from the given path and filtering out texts that have length greater than
    `max_text_len` and images that have width greater than `img_width`. The remaining strokes and images are then padded
    with zeros to reach the specified `max_seq_len` and `img_height`.

    Args:
        path (str): path to the data file to be loaded;
        max_text_len (int): maximum length of the text in the input data;
        max_seq_len (int): maximum length of the strokes in the input data;
        img_width (int): maximum width of the images in the input data;
        img_height (int): height of the images in the input data.

    Returns:
        Tuple[List[np.ndarray], np.ndarray, np.ndarray]: tuple containing the list of strokes, the array of texts and the array of images.
    """
    with open(path, "rb") as f:
        ds = pickle.load(f)

    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len - len(text),))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            if x is not None and sample.shape[1] < img_width:
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype("int32")
    samples = np.array(samples)
    return strokes, texts, samples


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

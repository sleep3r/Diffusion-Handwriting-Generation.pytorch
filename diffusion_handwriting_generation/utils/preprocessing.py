import os
import pickle
import random

import numpy as np

from diffusion_handwriting_generation.config import DLConfig, config_entrypoint
from diffusion_handwriting_generation.tokenizer import Tokenizer

"""
Creates the online and offline dataset for training
Before running this script, download the following things from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database
data/lineStrokes-all.tar.gz   -   the stroke xml for the online dataset
data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset
extract these contents and put them in the ./data directory (unless otherwise specified)
they should have the same names, e.g. "lineStrokes-all" (unless otherwise specified)
"""


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
        img (np.ndarray): input image, shape (`height`, `W`);
        width (int): desired width of the padded image;
        height (int): desired height of the padded image.

    Returns:
        np.ndarray: padded image, shape (`height`, `width`).
    """
    pad_len = width - img.shape[1]
    whites = np.ones((height, pad_len)) * 255
    img = np.concatenate((img, whites), axis=1).astype("float32")
    return img


def remove_whitespace(img, thresh, remove_middle=False):
    # removes any column or row without a pixel less than specified threshold
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)

    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle:
        return img[rows[0]][:, cols[0]]
    else:
        rows, cols = rows[0], cols[0]
        return img[rows[0] : rows[-1], cols[0] : cols[-1]]



def main(cfg: DLConfig) -> None:
    pass


if __name__ == "__main__":
    config: DLConfig = config_entrypoint()
    main(config)

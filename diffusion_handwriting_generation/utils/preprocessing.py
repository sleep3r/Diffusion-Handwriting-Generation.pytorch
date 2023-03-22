import numpy as np


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
    return np.concatenate((x, padding)).astype("float32")


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
    return np.concatenate((img, whites), axis=1).astype("float32")


def remove_whitespace(
    img: np.ndarray,
    thresh: float,
    remove_middle: bool = False,
) -> np.ndarray:
    # removes any column or row without a pixel less than specified threshold
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)

    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle:
        return img[rows[0]][:, cols[0]]
    rows, cols = rows[0], cols[0]
    return img[rows[0] : rows[-1], cols[0] : cols[-1]]

import xml.etree.ElementTree as ET
from os import PathLike
from pathlib import Path

import cv2
import numpy as np

from diffusion_handwriting_generation.utils.preprocessing import remove_whitespace


def parse_strokes_xml(xml_path: PathLike | str) -> np.ndarray:
    """Parses an XML strokes file from the IAM Handwriting Database and returns a list of strokes.

    Matches TensorFlow implementation semantics:
    - Pen-lift channel is ROLLED by 1 position (so a stroke segment has pen_lift=1 if it should NOT be drawn)
    - No sorting by timestamp (preserves file order)
    - is_end is marked at the last point of each stroke, then rolled

    Args:
        xml_path (str): Path to the XML strokes file.

    Returns:
        np.ndarray: shape (num_points, 3), where last dimension is (dx, dy, pen_lift)
    """
    # Parse the XML file using ElementTree
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the StrokeSet element
    stroke_set = root.find("StrokeSet")
    if stroke_set is None:
        raise ValueError("No StrokeSet element found in XML file")

    # Extract the strokes from the StrokeSet element - match TF semantics
    strokes = []
    prev = None

    for stroke_elem in stroke_set.findall("Stroke"):
        points = stroke_elem.findall("Point")
        for idx, point in enumerate(points):
            x = int(point.attrib["x"])
            y = int(point.attrib["y"])
            # Mark as end if this is the last point in the stroke
            is_end = 1.0 if idx == len(points) - 1 else 0.0

            if prev is None:
                prev = [x, -y]
            else:
                strokes.append([x - prev[0], -y - prev[1], is_end])
                prev = [x, -y]

    strokes = np.array(strokes, dtype=float)

    # CRITICAL: Roll pen-lift channel by 1 (TF semantics)
    # "the pen pickups are shifted by one, so a stroke that is not drawn has a 1"
    strokes[:, 2] = np.roll(strokes[:, 2], 1)

    # Normalize the strokes
    strokes[:, :2] /= np.std(strokes[:, :2])

    # Apply stroke simplification (combine_strokes) 3 times, matching TF
    # This reduces micro-jitter and helps learning
    for _ in range(3):
        strokes = combine_strokes(strokes, int(len(strokes) * 0.2))

    return strokes


def parse_lines_txt(ascii_file: Path) -> dict:
    """
    Parses an ASCII lines file from the IAM Handwriting Database.

    Args:
        ascii_file (Path): path to the ASCII lines file.

    Returns:
        dict: dictionary with the line ID as key and the text as value.
    """
    texts = {}
    has_started = False
    lines_num = -1

    with ascii_file.open("r") as f:
        for line in f.readlines():
            if "CSR" in line:
                has_started = True
            # the text under 'CSR' is correct, the one labeled under 'OCR' is not

            if has_started:
                if lines_num > 0:  # there is one space after 'CSR'
                    if line.strip():  # if the line is not empty
                        texts[f"{ascii_file.stem}-{lines_num:02d}"] = line[:-1]

                lines_num += 1
    return texts


def read_img(path: PathLike | str, height: int) -> np.ndarray:
    """
    Loads an image from the given path, removes white spaces and resizes it to the given height.

    Args:
        path (str): path to the image;
        height (int): desired height of the image.

    Returns:
        np.ndarray: processed image
    """
    if isinstance(path, PathLike):
        path = str(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = remove_whitespace(img, thresh=127)
    h, w = img.shape
    return cv2.resize(img, (height * w // h, height), interpolation=cv2.INTER_CUBIC)


def combine_strokes(x: np.ndarray, n: int) -> np.ndarray:
    """
    Consecutive stroke vectors who point in similar directions are summed
    if the pen was picked up in either of the strokes,
    we pick up the pen for the combined stroke.

    Args:
        x (np.ndarray): strokes;
        n (int): number of strokes to combine.

    Returns:
        np.ndarray: combined strokes.
    """

    def norms(x):
        return np.linalg.norm(x, axis=1)

    s, s_neighbors = x[::2, :2], x[1::2, :2]

    if len(x) % 2 != 0:
        s = s[:-1]

    values = norms(s) + norms(s_neighbors) - norms(s + s_neighbors)
    ind = np.argsort(values)[:n]

    x[ind * 2] += x[ind * 2 + 1]
    x[ind * 2, 2] = np.greater(x[ind * 2, 2], 0)
    x = np.delete(x, ind * 2 + 1, axis=0)
    x[:, :2] /= np.std(x[:, :2])
    return x

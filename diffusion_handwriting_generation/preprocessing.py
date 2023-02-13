import os
import pickle
import random

import fire
import numpy as np

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


def norms(x):
    return np.linalg.norm(x, axis=-1)


def combine_strokes(x, n):
    # consecutive stroke vectors who point in similar directions are summed
    # if the pen was picked up in either of the strokes,
    # we pick up the pen for the combined stroke
    s, s_neighbors = x[::2, :2], x[1::2, :2]
    if len(x) % 2 != 0:
        s = s[:-1]
    values = norms(s) + norms(s_neighbors) - norms(s + s_neighbors)
    ind = np.rt(values)[:n]
    x[ind * 2] += x[ind * 2 + 1]
    x[ind * 2, 2] = np.greater(x[ind * 2, 2], 0)
    x = np.delete(x, ind * 2 + 1, axis=0)
    x[:, :2] /= np.std(x[:, :2])
    return x


def parse_page_text(dir_path, id):
    dict = {}
    f = open(dir_path + "/" + id)
    has_started = False
    line_num = -1
    for l in f.readlines():
        if "CSR" in l:
            has_started = True
        # the text under 'CSR' is correct, the one labeled under 'OCR' is not
        if has_started:
            if line_num > 0:  # there is one space after 'CSR'
                dict[id[:-4] + "-%02d" % line_num] = l[:-1]
                # add the id of the line -0n as a key to dictionary,
                # with value of the line number (excluding the last \n)
            line_num += 1
    return dict


def create_dict(path):
    # creates a dictionary of all the line IDs and their respective texts
    dict = {}
    for dir in os.listdir(path):
        dirpath = path + "/" + dir
        for subdir in os.listdir(dirpath):
            subdirpath = dirpath + "/" + subdir
            forms = os.listdir(subdirpath)
            [dict.update(parse_page_text(subdirpath, f)) for f in forms]
    return dict


def parse_stroke_xml(path):
    xml = open(path)
    xml = xml.readlines()
    strokes = []
    previous = None
    for i, l in enumerate(xml):
        if "Point" in l:
            x_ind, y_ind, y_end = l.index("x="), l.index("y="), l.index("time=")
            x = int(l[x_ind + 3 : y_ind - 2])
            y = int(l[y_ind + 3 : y_end - 2])
            is_end = 1.0 if "/Stroke" in xml[i + 1] else 0.0
            if previous is None:
                previous = [x, -y]
            else:
                strokes.append([x - previous[0], -y - previous[1], is_end])
                previous = [x, -y]

    strokes = np.array(strokes)
    strokes[:, 2] = np.roll(strokes[:, 2], 1)
    # currently, a stroke has a 1 if the next stroke is not drawn
    # the pen pickups are shifted by one, so a stroke that is not drawn has a 1
    strokes[:, :2] /= np.std(strokes[:, :2])
    for i in range(3):
        strokes = combine_strokes(strokes, int(len(strokes) * 0.2))
    return strokes


import cv2


def read_img(path: str, height: int) -> np.ndarray:
    """
    Loads an image from the given path, removes white spaces and resizes it to the given height.


        path (str): path to the image;
        height (int): desired height of the image.

    Returns:
        np.ndarray: processed image
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = remove_whitespace(img, thresh=127)
    h, w = img.shape
    img = cv2.resize(img, (height * w // h, height), interpolation=cv2.INTER_CUBIC)
    return img


def create_dataset(formlist, strokes_path, images_path, tokenizer, text_dict, height):
    dataset = []
    with open(formlist) as f:
        forms = f.readlines()

    for f in forms:
        path = strokes_path + "/" + f[1:4] + "/" + f[1:8]
        offline_path = images_path + "/" + f[1:4] + "/" + f[1:8]

        samples = [s for s in os.listdir(path) if f[1:-1] in s]
        offline_samples = [s for s in os.listdir(offline_path) if f[1:-1] in s]
        shuffled_offline_samples = offline_samples.copy()
        random.shuffle(shuffled_offline_samples)

        for i in range(len(samples)):
            dataset.append(
                (
                    parse_stroke_xml(path + "/" + samples[i]),
                    tokenizer.encode(text_dict[samples[i][:-4]]),
                    read_img(offline_path + "/" + shuffled_offline_samples[i], height),
                )
            )
    return dataset


def main(
    text_path: str | None = "./data/ascii-all",
    strokes_path: str | None = "./data/lineStrokes-all",
    images_path: str | None = "./data/lineImages-all",
    height: int | None = 96,
) -> None:
    """
    Main function to run the script.


        text_path (str, optional): path to text labels, defaults to "./data/ascii-all";
        strokes_path (str, optional): path to stroke xml, defaults to "./data/lineStrokes-all";
        images_path (str, optional): path to line images, defaults to "./data/lineImages-all";
        height (int, optional): height of offline images, defaults to 96.

    Returns:
        None
    """
    t_path = text_path
    s_path = strokes_path
    i_path = images_path
    H = height

    train_info = "./data/trainset.txt"
    val1_info = "./data/testset_f.txt"  # labeled as test, valid set 1 as test instead
    val2_info = "./data/testset_t.txt"
    test_info = "./data/testset_v.txt"  # labeled as valid, but we use as test

    tok = Tokenizer()
    labels = create_dict(t_path)
    train_strokes = create_dataset(train_info, s_path, i_path, tok, labels, H)
    val1_strokes = create_dataset(val1_info, s_path, i_path, tok, labels, H)
    val2_strokes = create_dataset(val2_info, s_path, i_path, tok, labels, H)
    test_strokes = create_dataset(test_info, s_path, i_path, tok, labels, H)

    train_strokes += val1_strokes
    train_strokes += val2_strokes
    random.shuffle(train_strokes)
    random.shuffle(test_strokes)

    with open("./data/train_strokes.p", "wb") as f:
        pickle.dump(train_strokes, f)
    with open("./data/test_strokes.p", "wb") as f:
        pickle.dump(test_strokes, f)


if __name__ == "__main__":
    fire.Fire(main)

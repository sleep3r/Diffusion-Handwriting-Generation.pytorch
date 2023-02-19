import copy
import json
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from diffusion_handwriting_generation.text_style import StyleExtractor
from diffusion_handwriting_generation.tokenizer import Tokenizer
from diffusion_handwriting_generation.utils.io import (
    parse_lines_txt,
    parse_strokes_xml,
    read_img,
)
from diffusion_handwriting_generation.utils.preprocessing import pad_img, pad_stroke_seq


class IAMDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        kind: Literal["train", "val", "test"] = "train",
        img_height: int = 96,
        img_width: int = 1500,
        max_text_len: int = 50,
        max_seq_len: int = 500,
        splits_file: PathLike | str = "splits.json",
        max_files: int | None = None,
    ):
        self.data_path = Path(data_dir)
        self.kind = kind
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len
        self.max_seq_len = max_seq_len

        self.max_files = max_files

        self.ascii_dir = self.data_path / "ascii"
        self.img_path = self.data_path / "lineImages"
        self.strokes_path = self.data_path / "lineStrokes"
        self.splits_file = splits_file
        with open(self.splits_file) as f:
            self.splits = json.load(f)

        self.tokenizer = Tokenizer()
        self.style_extractor = StyleExtractor()

        self.__init_dataset()

    def __len__(self):
        return len(self.dataset)

    @property
    def dataset(self) -> list[dict]:
        return self._dataset

    def __init_dataset(self) -> None:
        dataset = []

        for f in tqdm(self.splits[self.kind]):
            strokes_path = self.strokes_path / f[:3] / f[:7]
            img_path = self.img_path / f[:3] / f[:7]

            text_dict = parse_lines_txt(self.ascii_dir / f[:3] / f[:7] / f"{f}.txt")

            for sample, text in text_dict.items():
                if len(text) > self.max_text_len:
                    continue

                raw_text = copy.deepcopy(text)

                strokes = parse_strokes_xml(strokes_path / f"{sample}.xml")
                strokes = pad_stroke_seq(strokes, maxlength=self.max_seq_len)

                text = self.tokenizer.encode(text)
                zeros_text = np.zeros((self.max_text_len - len(text),))
                text = np.concatenate((text, zeros_text))

                img = read_img(img_path / f"{sample}.tif", self.img_height)

                if strokes is not None and img.shape[1] < self.img_width:
                    img = pad_img(img, self.img_width, self.img_height)
                    style = self.style_extractor(img[None, None, :])

                    dataset.append(
                        {
                            "sample": sample,
                            "strokes": strokes,
                            "text": text,
                            "image": img,
                            "style": style,
                            "raw_text": raw_text,
                        }
                    )

                    if self.max_files and len(dataset) >= self.max_files:
                        self._dataset = dataset
                        return

        self._dataset = dataset

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "strokes": torch.FloatTensor(self.dataset[idx]["strokes"]),
            "text": torch.IntTensor(self.dataset[idx]["text"]),
            "style": self.dataset[idx]["style"],
        }


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

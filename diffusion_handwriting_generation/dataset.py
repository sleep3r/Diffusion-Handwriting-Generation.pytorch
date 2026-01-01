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
        kind: Literal["train", "validation", "test"] = "train",
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
        import random

        dataset = []

        for f in tqdm(self.splits[self.kind], desc="Loading dataset"):
            strokes_dir = self.strokes_path / f[:3] / f[:7]
            img_dir = self.img_path / f[:3] / f[:7]
            ascii_path = self.ascii_dir / f[:3] / f[:7] / f"{f}.txt"

            if not ascii_path.exists():
                continue

            text_dict = parse_lines_txt(ascii_path)

            # 1. Collect valid samples for this form (writer)
            form_valid_samples = []
            for sample_id, text in text_dict.items():
                if len(text) >= self.max_text_len:
                    continue

                # Check files exist
                if not (strokes_dir / f"{sample_id}.xml").exists():
                    continue
                if not (img_dir / f"{sample_id}.tif").exists():
                    continue

                form_valid_samples.append(sample_id)

            # 2. Process samples and assign randomized style
            for sample_id in form_valid_samples:
                text = text_dict[sample_id]

                # Parse strokes
                strokes = parse_strokes_xml(strokes_dir / f"{sample_id}.xml")
                strokes = pad_stroke_seq(strokes, maxlength=self.max_seq_len)  # type: ignore

                if strokes is None:
                    continue

                # Encode text
                encoded_text = self.tokenizer.encode(text)
                zeros_text = np.zeros((self.max_text_len - len(encoded_text),))
                encoded_text = np.concatenate((encoded_text, zeros_text))

                # Load content image (for reference/viz, usually not used by model directly if style is separate)
                img = read_img(img_dir / f"{sample_id}.tif", self.img_height)

                # CRITICAL: Pick style source - random DIFFERENT sample from same form/writer
                # This prevents the model from "reading" the text from the style vector
                style_source_id = sample_id
                if len(form_valid_samples) > 1:
                    candidates = [sid for sid in form_valid_samples if sid != sample_id]
                    style_source_id = random.choice(candidates)

                # Load style image
                style_img = read_img(img_dir / f"{style_source_id}.tif", self.img_height)

                if img.shape[1] < self.img_width:
                    img = pad_img(img, self.img_width, self.img_height)

                    # Pad style image if needed
                    if style_img.shape[1] < self.img_width:
                        style_img = pad_img(style_img, self.img_width, self.img_height)

                    # Extract style from the RANDOMIZED source
                    # StyleExtractor is now in .eval() mode, so this is stable
                    # Move to CPU immediately to avoid pinning errors in DataLoader
                    style = self.style_extractor(style_img[None, None, :])[
                        0
                    ].cpu()  # Remove batch dim and move to CPU

                    dataset.append(
                        {
                            "sample": sample_id,
                            "strokes": strokes,
                            "text": encoded_text,
                            "image": img,
                            "style": style,
                            "raw_text": text,
                            "style_source": style_source_id,  # Track source for debugging
                        },
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

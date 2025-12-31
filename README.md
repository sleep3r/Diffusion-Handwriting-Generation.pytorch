<div align="center">

<img src="https://raw.githubusercontent.com/sleep3r/Diffusion-Handwriting-Generation.pytorch/main/pic.png" style="height: auto; width: 70%;">

<b>Diffusion Model for Handwriting Generation</b>

[![CodeFactor](https://www.codefactor.io/repository/github/sleep3r/Diffusion-Handwriting-Generation.pytorch/badge)](https://www.codefactor.io/repository/github/sleep3r/Diffusion-Handwriting-Generation.pytorch)
[![python](https://img.shields.io/badge/python_3.10-passing-success)](https://github.com/sleep3r/Diffusion-Handwriting-Generation.pytorch/badge.svg?branch=main&event=push)
</div>

----

## Data preparation:

Download [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and extract it
to the `data/` directory with the following structure:

```
data/
├── ascii/
│   ├── forms.txt
│   ├── lines.txt
│   └── sentences.txt
├── lineImages/
│   └── (png files organized by form)
└── lineStrokes/
    └── (xml files organized by form)
```

After extracting the data, run the data preparation script to create individual form text files:

```bash
uv run python prepare_data.py
```

This will parse `lines.txt` and create structured text files in `data/ascii/` directory (e.g., `data/ascii/a01/a01-000u/a01-000u.txt`).

## Install:

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install
```

## Train:
First, configure your training in `configs/<cfg>.yml`:

```yml
experiment:
  name: <exp_name>
  work_dir: <work_dir>
  data_dir: <data_dir>
```

Then, run:

```bash
make train CONFIG=<cfg>.yml
```

## References:

|Papers|
|---|
| [[1]](https://arxiv.org/abs/2011.06704) Luhman, Troy, and Eric Luhman. "Diffusion models for handwriting generation." arXiv preprint arXiv:2011.06704 (2020). | 
<div align="center">

<img src="https://raw.githubusercontent.com/sleep3r/Diffusion-Handwriting-Generation.pytorch/main/pic.png" style="height: auto; width: 70%;">

<b>Diffusion Model for Handwriting Generation</b>

[![CodeFactor](https://www.codefactor.io/repository/github/sleep3r/Diffusion-Handwriting-Generation.pytorch/badge)](https://www.codefactor.io/repository/github/sleep3r/Diffusion-Handwriting-Generation.pytorch)
[![python](https://img.shields.io/badge/python_3.10-passing-success)](https://github.com/sleep3r/Diffusion-Handwriting-Generation.pytorch/badge.svg?branch=main&event=push)
</div>

----

## Data preparation

### 1. Download IAM Handwriting Database

You need to download the following files from [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (registration required):

- **ascii.tgz** - Contains forms.txt, lines.txt, sentences.txt
- **lineImages.tgz** - Line images as PNG files  
- **lineStrokes-all.tar.gz** - Individual XML files with stroke data for each line ⚠️ **Important:** Download `lineStrokes-all.tar.gz`, NOT `forms.tgz`

### 2. Extract files

Extract the downloaded archives to the `data/` directory with the following structure:

```bash
data/
├── ascii/
│   ├── forms.txt
│   ├── lines.txt
│   └── sentences.txt
├── lineImages/
│   └── a01/a01-000u/
│       ├── a01-000u-00.png
│       ├── a01-000u-01.png
│       └── ...
└── lineStrokes/
    └── a01/a01-000u/
        ├── a01-000u-00.xml  (with StrokeSet data)
        ├── a01-000u-01.xml
        └── ...
```

**Important:** `lineStrokes/` must contain individual XML files for each line (e.g., `a01-000u-00.xml`) with `<StrokeSet>` elements containing stroke points, NOT form-level metadata XML files.

### 3. Prepare text files

After extracting the data, run the data preparation script to create individual form text files:

```bash
make prepare_data
```

This will parse `lines.txt` and create structured text files in `data/ascii/` directory (e.g., `data/ascii/a01/a01-000u/a01-000u.txt`).

## Install

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install
```

## Train

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

## References

|Papers|
|---|
| [[1]](https://arxiv.org/abs/2011.06704) Luhman, Troy, and Eric Luhman. "Diffusion models for handwriting generation." arXiv preprint arXiv:2011.06704 (2020). | 
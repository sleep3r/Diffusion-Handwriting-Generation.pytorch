import string

import numpy as np
import torch


class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = "_" + string.ascii_letters + string.digits + ".?!,'\"- "
        self.numbers = np.arange(2, len(self.text) + 2)
        self.create_dict()
        self.vocab_size = len(self.text) + 2

    def create_dict(self):
        for (
            char,
            token,
        ) in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = " ", "<end>"  # only for decoding

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text:
                tokenized.append(self.tokens[char])
            else:
                tokenized.append(2)  # unknown character is '_', which has index 2

        tokenized.append(1)  # 1 is the end of sentence character
        return tokenized

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return "".join(text)

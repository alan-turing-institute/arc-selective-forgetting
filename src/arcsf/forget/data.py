"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/data_module.py
"""

import torch
from torch.utils.data import Dataset


def get_uniform_selector(seed=None):
    if seed is not None:
        gen = torch.random.manual_seed(seed)
    else:
        gen = torch.random.default_generator

    def uniform_selector(idx, N):
        return torch.randint(0, N, (1,), generator=gen).item()

    return uniform_selector


class ForgetDataset(Dataset):
    def __init__(
        self,
        forget_data,
        tokenizer,
        retain_data=None,
        idk_processor=None,
        retain_selector=get_uniform_selector(),
        text_key="text",
    ):
        super().__init__()
        self.forget_data = forget_data
        self.tokenizer = tokenizer
        self.retain_data = retain_data
        self.idk_processor = idk_processor
        self.retain_selector = retain_selector
        self.text_key = text_key

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        forget_inputs = self.forget_data[idx][self.text_key]
        if self.retain_data:
            retain_idx = self.retain_selector(idx, len(self.retain_data))
            retain_inputs = self.retain_data[retain_idx][self.text_key]
        else:
            retain_inputs = None
        idk_inputs = self.idk_processor(forget_inputs) if self.idk_processor else None

        forget_inputs = self.tokenizer(forget_inputs)
        retain_inputs = self.tokenizer(retain_inputs) if retain_inputs else None
        idk_inputs = self.tokenizer(idk_inputs) if idk_inputs else None

        return forget_inputs, retain_inputs, idk_inputs

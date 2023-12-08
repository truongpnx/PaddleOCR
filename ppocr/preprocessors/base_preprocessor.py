import torch.nn as nn


class BasePreprocessor(nn.Module):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x

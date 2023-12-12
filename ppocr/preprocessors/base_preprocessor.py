import paddle.nn as nn


class BasePreprocessor(nn.Layer):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x

"""
Utilities for working with tensors.

"""

from typing import Union

import numpy as np
import torch

__all__ = ['process_tensor']


def process_tensor(img_tensor: torch.Tensor, normalize=True) -> np.ndarray:
    """
    Converts a tensor to a Numpy array and normalize it to [0, 1].

    Args:
        img_tensor: A tensor of shape (C, H, W).
        normalize: A flag that decides whether the output should be normalized or not.

    Returns:
        A Numpy array of shape (H, W, C).

    """
    img_numpy = img_tensor.detach().cpu().numpy()
    img_numpy = np.squeeze(img_numpy)
    img_numpy = np.transpose(img_numpy, (1, 2, 0))
    
    if normalize:
        # Normalizes pixel values to [0, 1.0]
        img_range = img_numpy.max() - img_numpy.min()
        if not np.isclose(img_range, 0, rtol=0, atol=1e-5):
            img_numpy = (img_numpy - img_numpy.min()) / img_range
    return img_numpy

"""
Utilities for working with images.

"""

from typing import Union, Tuple

import numpy as np
import torch

__all__ = ['normalize_img', 'one_sided_zero_pad']


def normalize_img(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Normalizes pixel values to [0.0, 1.0]."""
    img_range = img.max() - img.min()
    output = img.copy()
    if not np.isclose(img_range, 0, rtol=0, atol=1e-5):
        output = (output - output.min()) / img_range
    return output

def one_sided_zero_pad(patch: np.ndarray, desired_size: int, box: Tuple[int, int, int, int]):
    """
    Return original patch if it is the right size. Assumes that the patch
    given is always smaller or equal to the desired size. The box tells us
    the spatial location of the patch on the image.
    """
    if len(patch.shape) != 3 or patch.shape[0] != 3:
        raise ValueError(f"patch must be have shape (3, height, width), but got {patch.shape}")
    if patch.shape[1] == desired_size and patch.shape[2] == desired_size:
        return patch

    vx_min, hx_min, vx_max, hx_max = box
    touching_top_edge = (vx_min <= 0)
    touching_left_edge = (hx_min <= 0)

    padded_patch = np.zeros((3, desired_size, desired_size))
    _, patch_h, patch_w = patch.shape

    if touching_top_edge and touching_top_edge:
        padded_patch[:, -patch_h:, -patch_w:] = patch  # fill from bottom right
    elif touching_top_edge:
        padded_patch[:, -patch_h:, :patch_w] = patch  # fill from bottom left
    elif touching_left_edge:
        padded_patch[:, :patch_h, -patch_w:] = patch  # fill from top right
    else:
        padded_patch[:, :patch_h, :patch_w] = patch  # fill from top left

    return padded_patch

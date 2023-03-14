"""
Making the .png files of top patch, which will later be updated to the AWS S3
bucket manually.

Logs
----
Model Name: alexnet
Optimization Method: SGD (no momentum)
Device: Macbook Pro 16" Late 2021 with M1 Pro
Time: 25 seconds
Space: generates about 185 MB of data.

Tony Fu, Bair Lab, March 2023

"""

#################################### GUARD ####################################

confirmation = input("Are you sure you want to run this code? (Y/N)")
if confirmation.lower() == "y":
    pass
else:
    print("Code execution aborted.")

###############################################################################


import os
import multiprocessing
from typing import Tuple

import torch
import numpy as np
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

from spatial_utils import SpatialIndexConverter
from model_utils import ModelInfo
from image_utils import one_sided_zero_pad, normalize_img

# Please specify some model details here:
MODEL_NAME = "alexnet"

# Set the result directory
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results', 'top_patch', MODEL_NAME)

########################### DON'T TOUCH CODE BELOW ############################

# Load model and related information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = getattr(models, MODEL_NAME)(pretrained=True).to(DEVICE)
MODEL_INFO = ModelInfo()
LAYER_NAMES = MODEL_INFO.get_layer_names(MODEL_NAME)

# Get the image directory
IMG_SIZE = (227, 227)
TOP_1 = 0
IMG_DIR = '/Users/tonyfu/Desktop/Bair Lab/top_and_bottom_images/images'
"""
Obviously, I cannot upload the content of IMG_DIR to GitHub because it is too
big. Here is some more information about the images at IMG_DIR so you can
try it yourself:

- Source: a subset ImageNet's testing set
- Number of image: 50000
- Size: 3 x 227 x 227
- Total size: about 62 GB
- Format: .npy (NumPy arrays)
- Note: The RGB values are batch-normalized to be roughly in the range of -1.0 to +1.0.
- Where you can download it: http://wartburg.biostr.washington.edu/loc/course/artiphys/data/i50k.html
"""

# Get the spatial indicies (for more info, see below)
def get_max_min_indicies(layer_name):
    spatial_index_path = os.path.join(CURRENT_DIR, os.pardir, "data",
                                      "top_100_image_patches",
                                      MODEL_NAME,f"{layer_name}.npy")
    return np.load(spatial_index_path).astype(int)
"""
I feel the need to explain what these spatial indices are all about. This was an idea that was developed by Dr. Wyeth Bair and his PhD student, Dr. Dean Pospisil. They trained neural networks on the 50,000 images mentioned earlier. As a reminder, the convolution (or more accurately, cross-correlation) operation slides the unit's kernel along the two spatial dimensions of the input. The first thing they did was to find the spatial locations that produced the maximum (most positive) responses. They repeated this process for all 50,000 images, and then ranked the resulting max locations to find out which image patches gave the strongest responses.

`MAX_MIN_INDICES` contains the results of this ranking for a particular convolutional layer of a model. The array has dimensions [num_units, 100, 4].

`num_units` represents the number of unique kernels in the convolutional layer. For example, Conv1 of AlexNet has 64 unique kernels. The second dimension `k` is 100 because the array stores the top and bottom 100 image patches. The last dimension has a size of 4 because it contains:
(1) `max_img_idx`: the index (ranging from 0 to 49,999) of the k-th most positive response image.
(2) `max_spatial_idx`: the spatial index of the kernel (not the pixel). The kernel is first slided along the x-axis, then y-axis. For instance, a spatial index of 0 corresponds to (0,0), and 1 to (0, 1). We can convert from 1D indexing to 2D using np.unravel_index(spatial_index, (output_height, output_width)).
(3) `min_img_idx`: same as `max_img_idx`, but for the k-th most negative response image.
(4) `min_spatial_idx`: same as `max_spatial_idx`, but for the k-th most negative response image patch.
"""

# Initiate helper objects. This object converts the spatial index from the
# output layer to that of the input layer (i.e., pixel coordinates).
converter = SpatialIndexConverter(MODEL, IMG_SIZE)


##################### Define a few small helper functions #####################

def clip(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def pad_box(box: Tuple[int, int, int, int], padding: int):
    """Makes sure box does not go beyond the image after padding."""
    y_min, x_min, y_max, x_max = box
    new_y_min = clip(y_min-padding, 0, IMG_SIZE[0])
    new_x_min = clip(x_min-padding, 0, IMG_SIZE[1])
    new_y_max = clip(y_max+padding, 0, IMG_SIZE[0])
    new_x_max = clip(x_max+padding, 0, IMG_SIZE[1])
    return new_y_min, new_x_min, new_y_max, new_x_max

###############################################################################

def save_image_patch_for_layer(layer_name):
    # Determine layer-specific information
    num_units = MODEL_INFO.get_num_units(MODEL_NAME, layer_name)
    layer_index = MODEL_INFO.get_layer_index(MODEL_NAME, layer_name)
    xn = MODEL_INFO.get_xn(MODEL_NAME, layer_name)
    rf_size = MODEL_INFO.get_rf_size(MODEL_NAME, layer_name)
    padding = (xn - rf_size) // 2


    # Define the output directory, create it if necessary
    layer_dir = os.path.join(RESULT_DIR, layer_name)
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)

    # Find the top- and bottom-100 image patches ranking of the layer
    max_min_indicies = get_max_min_indicies(layer_name)

    
    for unit_index in tqdm(range(num_units)):
        # Get top and bottom image indices and patch spatial indices
        max_n_img_index   = max_min_indicies[unit_index, TOP_1, 0]
        max_n_patch_index = max_min_indicies[unit_index, TOP_1, 1]

        # Convert from output spatial index to pixel coordinate
        box = converter.convert(max_n_patch_index, layer_index, 0, is_forward=False)
        
        # Prevent indexing out of range
        y_min, x_min, y_max, x_max = pad_box(box, padding)
        
        # Load the image
        img_path = os.path.join(IMG_DIR, f"{max_n_img_index}.npy")
        img_numpy = np.load(img_path)[:, y_min:y_max+1, x_min:x_max+1]
        
        # Pad it to (3, xn, xn) if necessary
        img_numpy = one_sided_zero_pad(img_numpy, xn, (y_min, x_min, y_max, x_max))
        img_numpy = normalize_img(img_numpy)
        
        plt.imshow(img_numpy.transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(layer_dir, f"{unit_index}.png"))
        plt.close()

if __name__ == '__main__':
    with multiprocessing.Pool(processes=len(LAYER_NAMES)) as pool:
        pool.map(save_image_patch_for_layer, LAYER_NAMES)

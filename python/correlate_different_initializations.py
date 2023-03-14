"""
Computes the Pearson coorelation coefficients of zero-intialized and top-patch-
initialized gradient ascent results.

Tony Fu, Bair Lab, March 2023

"""

import os

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from model_utils import ModelInfo

# Please specify some details here:
MODEL_NAME = "alexnet"
OPTIMIZATION_METHOD = 'SGD'

# Locate the result directories
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ZERO_INIT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results',
                             OPTIMIZATION_METHOD, 'zero_initialized', MODEL_NAME)
TOP_PATCH_INIT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results',
                             OPTIMIZATION_METHOD, 'top_patch_initialized', MODEL_NAME)

# Set the output path
OUTPUT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results',
                           OPTIMIZATION_METHOD, 'correlation')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_zero_vs_top_patch.txt")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

########################### DON'T TOUCH CODE BELOW ############################

# Load model and related information
MODEL_INFO = ModelInfo()
LAYER_NAMES = MODEL_INFO.get_layer_names(MODEL_NAME)

##################### Define a few small helper functions #####################

def crop_padding(img, padding):
    """Removes the padding from the image."""
    return img[padding:-padding, padding:-padding, :]

###############################################################################

with open(OUTPUT_PATH, "w") as f:
    f.write(f"layer_name unit_index correlation\n")
    
    for layer_name in LAYER_NAMES:
        # Determine layer-specific information
        print(f"Computing correlation for {layer_name}...")
        num_units = MODEL_INFO.get_num_units(MODEL_NAME, layer_name)
        xn = MODEL_INFO.get_xn(MODEL_NAME, layer_name)
        rf_size = MODEL_INFO.get_rf_size(MODEL_NAME, layer_name)
        padding = (xn - rf_size) // 2
        
        # Load the result arrays of two different initializations
        zero_init_layer_array = np.load(os.path.join(ZERO_INIT_DIR, layer_name, f"{layer_name}.npy"))
        top_patch_init_layer_array = np.load(os.path.join(TOP_PATCH_INIT_DIR, layer_name, f"{layer_name}.npy"))
        
        for unit_index in tqdm(range(num_units)):
            # Remove the surrounding zero padding
            img1 = crop_padding(zero_init_layer_array[unit_index], padding)
            img2 = crop_padding(top_patch_init_layer_array[unit_index], padding)

            # Compute and record the Pearson correlation coefficient
            correlation, _ = pearsonr(img1.flatten(), img2.flatten())
            f.write(f"{layer_name} {unit_index} {correlation:.4f}\n")

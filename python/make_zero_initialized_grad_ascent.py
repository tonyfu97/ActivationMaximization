"""
Making gradient ascent visualization.

WARNING: This script takes time and memory to run.

Logs
----
Model Name: alexnet
Optimization Method: SGD (no momentum)
Device: Macbook Pro 16" Late 2021 with M1 Pro
Time: 2 hours and 8 minutes
Space: generates about 85 MB of data.

Tony Fu, Bair Lab, March 2023

"""

# #################################### GUARD ####################################

# confirmation = input("Are you sure you want to run this code? (Y/N)")
# if confirmation.lower() == "y":
#     pass
# else:
#     print("Code execution aborted.")

# ###############################################################################


import os
import multiprocessing

import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom modules
from model_utils import ModelInfo, get_truncated_model
from tensor_utils import process_tensor
from grad_ascent import GradientAscent

# Specify the model and optimization method of interest
MODEL_NAME = 'alexnet'
OPTIMIZATION_METHOD = 'SGD'  # options: SGD and Adam

# Setting up
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = getattr(models, MODEL_NAME)(pretrained=True).to(DEVICE)
MODEL_INFO = ModelInfo()
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results',
                          OPTIMIZATION_METHOD, 'zero_initialized', MODEL_NAME)

# Compute Gradient Ascent visualizations and save them to .png
NUM_ITER = 100
LR = 0.1
MOMENTUM = False
LAYER_NAMES = MODEL_INFO.get_layer_names(MODEL_NAME)


def create_visualizations_for_layer(layer_name):
    # Get layer-specific information
    num_units = MODEL_INFO.get_num_units(MODEL_NAME, layer_name)
    xn = MODEL_INFO.get_xn(MODEL_NAME, layer_name)
    layer_index = MODEL_INFO.get_layer_index(MODEL_NAME, layer_name)
    truncated_model = get_truncated_model(MODEL, layer_index)
    print(f"Creating Gradient Ascent visualizations for {MODEL_NAME} {layer_name}...")
    
    # Create directory to store results
    layer_dir = os.path.join(RESULT_DIR, layer_name)
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    
    # We will also store the results in a numpy array
    result_array = np.zeros((num_units, xn, xn, 3))

    for unit_index in tqdm(range(num_units)):
        # Computer gradient ascent
        img = torch.zeros(1, 3, xn, xn, requires_grad=True, device=DEVICE)
        ga = GradientAscent(truncated_model, unit_index, img, lr=LR,
                            optimizer=OPTIMIZATION_METHOD, momentum=MOMENTUM)
        for i in range(NUM_ITER - 1):
            ga.step()
        result = ga.step()
        
        # Save result to an image
        result_array[unit_index] = process_tensor(result)
        plt.imshow(result_array[unit_index])
        plt.axis('off')
        plt.savefig(os.path.join(layer_dir, f"{unit_index}.png"))
        plt.close()
    
    np.save(os.path.join(layer_dir, f"{layer_name}.npy"), result_array)


if __name__ == '__main__':
    with multiprocessing.Pool(processes=len(LAYER_NAMES)) as pool:
        pool.map(create_visualizations_for_layer, LAYER_NAMES)

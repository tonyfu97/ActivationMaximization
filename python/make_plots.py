"""
Script to plot the distribution of correlation coefficients of each layer.

"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import ModelInfo

# Please specify some details here:
MODEL_NAME = "alexnet"
OPTIMIZATION_METHOD = 'SGD'

# Load correlations
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CORRELATION_PATH = os.path.join(CURRENT_DIR, os.pardir, 'results',
                                OPTIMIZATION_METHOD, 'correlation',
                                f"{MODEL_NAME}_zero_vs_top_patch.txt")
CORRELATIONS = pd.read_csv(CORRELATION_PATH, delim_whitespace=True)
CORRELATIONS.columns = ["layer_name", "unit_index", "correlation"]

# Define helper function
def get_layer_correlations(layer_name):
    return CORRELATIONS.loc[(CORRELATIONS["layer_name"] == layer_name), 'correlation']

# Get the name of layers
MODEL_INFO = ModelInfo()
LAYER_NAMES = MODEL_INFO.get_layer_names(MODEL_NAME)

# Plot the distribution of correlation coeff in each layer
bins = np.linspace(-1.1, 1.1, 50)
all_layer_corr = []
plt.figure(figsize=(15,3))
for i, layer_name in enumerate(LAYER_NAMES):
    layer_corr = get_layer_correlations(layer_name)
    all_layer_corr.append(layer_corr)

    plt.subplot(1,len(LAYER_NAMES),i+1)
    plt.hist(layer_corr, bins=bins)
    plt.title(layer_name)
    plt.xlim([-1.1, 1.1])
plt.show()

# Summarize the trend in a single plot, too.
plt.boxplot(all_layer_corr, labels=LAYER_NAMES)
plt.show()

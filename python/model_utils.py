"""
Utilities for working with pre-trained models.

Tony Fu, Bair Lab, Feb 2023

"""

import os
import copy

import pandas as pd
import torch.fx as fx
import torch.nn as nn

__all__ = ['ModelInfo', 'get_truncated_model']

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_INFO_FILE_PATH = os.path.join(CURRENT_DIR, os.pardir, "data", "model_info.txt")


class ModelInfo:
    """
    A class for loading and accessing model information from a CSV file.

    Attributes:
        model_info (pandas.DataFrame): The model information, loaded from the
        CSV file.

    """
    def __init__(self):
        """
        Initializes a new instance of the ModelInfo class.

        """
        self.model_info = self._load_model_info(MODEL_INFO_FILE_PATH)

    def _load_model_info(self, model_info_file_path: str) -> pd.DataFrame:
        """
        Loads the model info from the specified CSV file.

        Args:
            model_info_file_path (str): The path to the CSV file containing the
            model information.

        Returns:
            pandas.DataFrame: The model information, loaded from the CSV file.

        """
        model_info = pd.read_csv(model_info_file_path, delim_whitespace=True)
        model_info.columns = ["model", "layer", "layer_index", "rf_size", "xn", "num_units"]
        return model_info

    def get_layer_names(self, model_name: str) -> int:
        """
        Returns the names of all conv layers of the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            list of string: The names of layers in the model.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name)
                                   , 'layer']

    def get_layer_index(self, model_name: str, layer_name: str) -> int:
        """
        Returns the index of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The index of the specified layer in the model.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'layer_index'].iloc[0]

    def get_num_units(self, model_name: str, layer_name: str) -> int:
        """
        Returns the number of units in the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The number of units in the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'num_units'].iloc[0]

    def get_rf_size(self, model_name: str, layer_name: str) -> int:
        """
        Returns the receptive field size (without additional padding to ensure
        the center unit's RF is indeed centered) of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The receptive field size of the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'rf_size'].iloc[0]

    def get_xn(self, model_name: str, layer_name: str) -> int:
        """
        Returns the receptive field size (plus additional padding to ensure the
        center unit's RF is indeed centered) of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The receptive field size of the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'xn'].iloc[0]


def get_truncated_model(model: nn.Module, layer_index: int) -> nn.Module:
    """
    Creates a truncated version of a neural network. Helps saves computation
    time if we just working with the first few layers.

    Args:
        model (nn.Module): The neural network to be truncated.
        layer_index (int): The index of the last layer (inclusive) to be
        included in the truncated model.

    Returns:
        A truncated version of the neural network.

    Example:
        model = models.alexnet(pretrained=True)
        model_to_conv2 = get_truncated_model(model, 3)
        y = model(torch.ones(1, 3, 200, 200))
    """
    model = copy.deepcopy(model)

    # IMPORTANT!! Set the model to evaluation mode to ensure that the traced
    # graph matches the behavior of the original model
    model.eval()

    # Trace the model using the FX framework
    graph = fx.Tracer().trace(model)

    # Create a new graph to hold the truncated version of the model
    new_graph = fx.Graph()

    # Keep track of the number of layers processed so far
    layer_counter = 0

    # Create a dictionary to map values in the original graph to values in the
    # new graph
    value_remap = {}

    # Iterate over the nodes in the original graph
    for node in graph.nodes:
        # Copy the node to the new graph and add it to the value remapping dictionary
        value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

        # If the node represents a module (i.e., a layer as oppposed to a
        # container "layer")...
        if node.op == 'call_module':
            # ...get the layer object
            layer = model
            for level in node.target.split('.'):
                layer = getattr(layer, level)

            # If we've reached the desired layer index...
            if layer_counter == layer_index:
                new_graph.output(node)
                break

            layer_counter += 1

    # Create a new GraphModule that combines the original model with the
    # truncated graph
    return fx.GraphModule(model, new_graph)

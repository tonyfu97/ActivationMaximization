"""
Script to convert Pytorch models into ONNX format, which is then used in an
interactive web app.

"""

import os

import torch
import torchvision.models as models

from model_utils import get_truncated_model, ModelInfo


######################## Define some helper functions #########################

def export_model(model_name: str, rf_data: ModelInfo, export_dir: str) -> None:
    """Export the model as onnx file. One file per conv layer."""
    model_func = getattr(models, model_name)
    model = model_func(pretrained=True)
    
    layer_names = rf_data.get_layer_names(model_name)
    
    for layer_name in layer_names:
        layer_index = rf_data.get_layer_index(model_name, layer_name)
        xn = rf_data.get_xn(model_name, layer_name)

        truncated_model = get_truncated_model(model, layer_index)
        truncated_model.eval()

        dummy_input = torch.zeros((1, 3, xn, xn))
        print(dummy_input.shape)
        torch.onnx.export(truncated_model, dummy_input,
                          os.path.join(export_dir,f'{model_name}_{layer_name}.onnx'))

###############################################################################

if __name__ == "__main__":
    MODEL_NAME = 'alexnet'
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    EXPORT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'docs', 'onnx_files')
    rf_data = ModelInfo()
    export_model(MODEL_NAME, rf_data, EXPORT_DIR)
    """
    Note: The onnx files of some deep layers are too large (> 10 Mb) to make
    sense in a web app. Consider deleting them manually after running this script.
    """

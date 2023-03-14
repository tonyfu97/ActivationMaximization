"""
Code for getting information about spatial dimensions and manipulating spatial
indices.

Note: all code assumes that the y-axis points downward.

Note2: This code turned out to be incompatible with the latest torch version
(broken hooks). If possible, install the previous versions using the commands
below:
>>pip uninstall torch                                
>>pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0

Tony Fu, Bair Lab, July 2022

"""

import math
import copy
from typing import Tuple, Optional, Union, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx
from torchvision import models

__all__ = ['SpatialIndexConverter']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################.#######################################
#                                                                             #
#                             HOOK FUNCTION BASE                              #
#                                                                             #
###############################################################################
class HookFunctionBase:
    """
    A base class that register a hook function to all specified layer types
    (excluding all container types) in a given model. The child class must
    implement hook_function(). The child class must also call
    self.register_forward_hook_to_layers() by itself.
    """
    def __init__(self, model: nn.Module, layer_types: Tuple[nn.Module]):
        """
        Constructs a HookFunctionBase object.

        Parameters
        ----------
        model : torchvision.models
            The neural network.
        layer_types : tuple of torch.nn.Modules
            A tuple of the layer types you would like to register the forward
            hook to. For example, layer_types = (nn.Conv2d, nn.ReLU) means
            that all the Conv2d and ReLU layers will be registered with the
            forward hook.
        """
        self.model = copy.deepcopy(model)
        self.model.to(DEVICE)
        self.model.eval()
        self.layer_types = layer_types

    def hook_function(self, module: nn.Module, ten_in: torch.Tensor,
                      ten_out: torch.Tensor) -> None:
        raise NotImplementedError("Child class of HookFunctionBase must "
                                  "implement hookfunction(self, module, ten_in, ten_out)")

    def register_forward_hook_to_layers(self, layer: nn.Module) -> None:
        # If "model" is a leave node and matches the layer_type, register hook.
        if (len(list(layer.children())) == 0):
            if (isinstance(layer, self.layer_types)):
                layer.register_forward_hook(self.hook_function)

        # Otherwise (i.e.,the layer is a container type layer), recurse.
        else:
            for sublayer in layer.children():
                self.register_forward_hook_to_layers(sublayer)


#######################################.#######################################
#                                                                             #
#                               SIZE INSPECTOR                                #
#                                                                             #
###############################################################################
class SizeInspector(HookFunctionBase):
    """
    A class that empirically determines the input and output sizes of all
    layers. This class determines the indexing convention of the layers. The
    indexing follows the flow of data through the model and excludes all
    container-type layers. For example, the indexing of
    torchvision.models.alexnet() is:

          no. | layer name
        ------+-----------
           0  |   Conv1
           1  |   ReLU1
           2  |  MaxPool1
           3  |   Conv2
           4  |   ReLU2
             ...
          19  |   ReLU7
          20  |  Linear3

    To get the indexing information for any arbitrary model, use the syntax:
        inspector = SizeInspector(model, image_size)
        inspector.print_summary()
    """
    def __init__(self, model: nn.Module, image_shape: Tuple[int, int]):
        super().__init__(model, layer_types=(torch.nn.Module))
        self.image_shape = image_shape
        self.layers = []
        self.input_sizes = []
        self.output_sizes = []
        self.register_forward_hook_to_layers(self.model)
        
        self.model(torch.zeros((1,3,*image_shape)).to(DEVICE))

    def hook_function(self, module: nn.Module, ten_in: torch.Tensor, ten_out: torch.Tensor) -> None:
        if (isinstance(module, self.layer_types)):
            self.layers.append(module)
            self.input_sizes.append(ten_in[0].shape[1:])
            self.output_sizes.append(ten_out.shape[1:])

    def print_summary(self) -> None:
        for i, layer in enumerate(self.layers):
            print("---------------------------------------------------------")
            print(f"  layer no.{i}: {layer}")
            try:
                print(f"  input size: ({self.input_sizes[i][0]}, "\
                      f"{self.input_sizes[i][1]}, {self.input_sizes[i][2]})")
                print(f" output size: ({self.output_sizes[i][0]}, "
                      f"{self.output_sizes[i][1]}, {self.output_sizes[i][2]})")
            except:
                print(" This layer is not 2D.")


if __name__ == '__main__':
    model = models.alexnet()
    inspector = SizeInspector(model, (227, 227))
    inspector.print_summary()


#######################################.#######################################
#                                                                             #
#                                    CLIP                                     #
#                                                                             #
###############################################################################
def clip(x: float, x_min: float, x_max: float) -> float:
    """Limits x to be x_min <= x <= x_max."""
    x = min(x_max, x)
    x = max(x_min, x)
    return x


#######################################.#######################################
#                                                                             #
#                                  LayerNode                                  #
#                                                                             #
###############################################################################
class LayerNode:
    def __init__(self,
                 name: str,
                 layer: Optional[torch.nn.Module] = None,
                 parents: Tuple['LayerNode', ...] = (),
                 children: Tuple['LayerNode', ...] = (),
                 idx: Optional[int] = None) -> None:
        self.idx = idx
        self.name = name
        self.layer = layer
        self.parents = parents
        self.children = children

    def __repr__(self) -> str:
        return f"LayerNode '{self.name}' (idx = {self.idx})\n"\
               f"       parents  = {self.parents}\n"\
               f"       children = {self.children}"


#######################################.#######################################
#                                                                             #
#                                 MAKE_GRAPH                                  #
#                                                                             #
###############################################################################
def make_graph(truncated_model: Union[fx.graph_module.GraphModule, torch.nn.Module])-> Dict[str, LayerNode]:
    """
    Generate a directed, acyclic graph representation of the model.

    Parameters
    ----------
    truncated_model : UNION[fx.graph_module.GraphModule, torch.nn.Module]
        The neural network. Can be truncated or not.

    Returns
    -------
    nodes : dict
        key : the unique name of each operation performed on the input tensor.
        value : a LayerNode object containing the information about the
                operation.
    """
    # Make sure that the truncated_model is a GraphModule. 
    if not isinstance(truncated_model, fx.graph_module.GraphModule):
        truncated_model = copy.deepcopy(truncated_model)
        graph = fx.Tracer().trace(truncated_model.eval())
        truncated_model = fx.GraphModule(truncated_model, graph)

    nodes = {}
    idx_count = 0  # for layer indexing
    # Populate the nodes dictionary with the initialized Nodes.
    for node in truncated_model.graph.nodes:
        # Get the layer torch.nn object.
        if node.op == 'call_module':
            layer = truncated_model
            idx = idx_count
            idx_count += 1
            for level in node.target.split("."):
                layer = getattr(layer, level)
        else:
            layer = None
            idx = None

        # Get the name of the parents.
        parents = []
        for parent in node.args:
            if isinstance(parent, fx.node.Node):
                parents.append(parent.name)

        # Initialize Nodes.
        nodes[node.name] = LayerNode(node.name, layer, parents=tuple(parents),
                                     idx=idx)

    # Determine the children of the nodes.
    for node in truncated_model.graph.nodes:
        for parent in nodes[node.name].parents:
            existing_children = nodes[parent].children
            nodes[parent].children = (*existing_children, node.name)

    return nodes


if __name__ == '__main__':
    model = models.resnet18()
    model.eval()
    for layer in make_graph(model).values():
        print(layer)


#######################################.#######################################
#                                                                             #
#                           SPATIAL INDEX CONVERTER                           #
#                                                                             #
###############################################################################
class SpatialIndexConverter(SizeInspector):
    """
    A class containing the model- and image-shape-specific conversion functions
    of the spatial indices across different layers. Useful for receptive field
    mapping and other tasks that involve the mappings of spatial locations
    onto a different layer.
    """
    def __init__(self, model: nn.Module, image_shape: Tuple[int, int]):
        """
        Constructs a SpatialIndexConverter object.

        Parameters
        ----------
        model : torchvision.models
            The neural network.
        image_shape : tuple of ints
            (vertical_dimension, horizontal_dimension) in pixels.
        """
        super().__init__(model, image_shape)
        self.dont_need_conversion = (nn.Sequential,
                                    nn.ModuleList,
                                    nn.Sigmoid,
                                    nn.ReLU,
                                    nn.Tanh,
                                    nn.Softmax2d,
                                    nn.BatchNorm2d,
                                    nn.Dropout2d,)
        self.need_convsersion = (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)
        
        # Represent the model as a directed, acyclic graph stored in a dict
        self.graph_dict = make_graph(model)
        self.idx_to_node = {node.idx: name for name, node in self.graph_dict.items()}

    def _forward_transform(self, x_min: int, x_max: int, stride: int,
                           kernel_size: int, padding: int, max_size: int) -> Tuple[int, int]:
        x_min = math.floor((x_min + padding - kernel_size)/stride + 1)
        x_min = clip(x_min, 0, max_size)
        x_max = math.floor((x_max + padding)/stride)
        x_max = clip(x_max, 0, max_size)
        return x_min, x_max

    def _backward_transform(self, x_min: int, x_max: int, stride: int,
                            kernel_size: int, padding: int, max_size: int) -> Tuple[int, int]:
        x_min = (x_min * stride) - padding
        x_min = clip(x_min, 0, max_size)
        x_max = (x_max * stride) + kernel_size - 1 - padding
        x_max = clip(x_max, 0, max_size)
        return x_min, x_max

    def _one_projection(self, layer_index: int, vx_min: int, hx_min: int,
                        vx_max: int, hx_max: int, is_forward: bool) -> None:
        layer = self.layers[layer_index]

        # Check the layer types to determine if a projection is necessary.
        if isinstance(layer, self.dont_need_conversion):
            return vx_min, hx_min, vx_max, hx_max
        if isinstance(layer, nn.Conv2d) and (layer.dilation != (1,1)):
            raise ValueError("Dilated convolution is currently not supported by SpatialIndexConverter.")
        if isinstance(layer, nn.MaxPool2d) and (layer.dilation != 1):
            raise ValueError("Dilated max pooling is currently not supported by SpatialIndexConverter.")

        # Use a different max size and transformation function depending on the
        # projection direction.
        if is_forward:
            _, v_max_size, h_max_size = self.output_sizes[layer_index]
            transform = self._forward_transform
        else:
            _, v_max_size, h_max_size = self.input_sizes[layer_index]
            transform = self._backward_transform

        if isinstance(layer, self.need_convsersion):
            try:
                vx_min, vx_max = transform(vx_min, vx_max,
                                        layer.stride[0], layer.kernel_size[0],
                                        layer.padding[0], v_max_size)
                hx_min, hx_max = transform(hx_min, hx_max,
                                        layer.stride[1], layer.kernel_size[1],
                                        layer.padding[1], h_max_size)
                return vx_min, hx_min, vx_max, hx_max
            except:
                # Sometimes the layer attributes do not come in the form of
                # a tuple. 
                vx_min, vx_max = transform(vx_min, vx_max,
                                        layer.stride, layer.kernel_size,
                                        layer.padding, v_max_size)
                hx_min, hx_max = transform(hx_min, hx_max,
                                        layer.stride, layer.kernel_size,
                                        layer.padding, h_max_size)
                return vx_min, hx_min, vx_max, hx_max

        raise ValueError(f"{type(layer)} is currently not supported by SpatialIndexConverter.")

    def _process_index(self, index: Union[Tuple[int, int], int],
                       start_layer_index: int) -> Tuple[int, int]:
        """
        Make sure that the index is a tuple of two indices. Unravel from 1D
        to 2D indexing if necessary.

        Returns
        -------
        index : tuple of ints
            The spatial coordinates of the point of interest in 
            (vertical index, horizontal index) format. Note that the vertical
            index increases downward.
        """
        try:
            if len(index)==2:
                return index
            else:
                raise Exception
        
        except:
            _, output_height, output_width = self.output_sizes[start_layer_index]
            return np.unravel_index(index, (output_height, output_width))

    def _merge_boxes(self, box_list: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Merges the two boxes such that the new box can contain all the boxes
        in the box_list. Each box must be in (vx_min, hx_min, vx_max, hx_max)
        format.
        """
        return min([box[0] for box in box_list]),\
               min([box[1] for box in box_list]),\
               max([box[2] for box in box_list]),\
               max([box[3] for box in box_list])

    def _forward_convert(self, vx_min: int, hx_min: int, vx_max: int, hx_max: int,
                         start_layer_name: str, end_layer_name:str) -> Tuple[int, int, int, int]:
        # If this 'start_layer_name' is a layer (as opposed to an operation),
        # calculate the new box.
        layer_index = self.graph_dict[start_layer_name].idx
        if isinstance(layer_index, int):
            vx_min, hx_min, vx_max, hx_max = self._one_projection(layer_index,
                                vx_min, hx_min, vx_max, hx_max, is_forward=True)

        # Base case:
        if start_layer_name == end_layer_name:
            return vx_min, hx_min, vx_max, hx_max

        # Recurse case:
        children = self.graph_dict[start_layer_name].children
        boxes = []
        for child in children:
            boxes.append(self._forward_convert(vx_min, hx_min, vx_max, hx_max, child, end_layer_name))
        return self._merge_boxes(boxes)

    def _backward_convert(self, vx_min: int, hx_min: int, vx_max: int, hx_max: int,
                          start_layer_name: str, end_layer_name: str) -> Tuple[int, int, int, int]:
        # If this 'start_layer_name' is a layer (as opposed to an operation),
        # calculate the new box.
        layer_index = self.graph_dict[start_layer_name].idx
        if isinstance(layer_index, int):
            vx_min, hx_min, vx_max, hx_max = self._one_projection(layer_index,
                                vx_min, hx_min, vx_max, hx_max, is_forward=False)

        # Base case:
        if start_layer_name == end_layer_name:
            return vx_min, hx_min, vx_max, hx_max

        # Recurse case:
        parents = self.graph_dict[start_layer_name].parents
        boxes = []
        for parent in parents:
            # Recurse case:
            boxes.append(self._backward_convert(vx_min, hx_min, vx_max, hx_max, parent, end_layer_name))
        return self._merge_boxes(boxes)

    def convert(self, index: Union[int, Tuple[int, int]],
                start_layer_index: int, end_layer_index: int,
                is_forward: bool) -> Tuple[int, int, int, int]:
        """
        Converts the spatial index across layers. Given a spatial location, the
        method returns a "box" in (vx_min, hx_min, vx_max, hx_max) format.

        Parameters
        ----------
        index : int or tuple of two ints
            The spatial index of interest. If only one int is provided, the
            function automatically unravel it accoording to the image's shape.
        start_layer_index : int
            The index of the starting layer. If you are not sure what index
            to use, call the .print_summary() method.
        end_layer_index : int
            The index of the destination layer. If you are not sure what index
            to use, call the .print_summary() method.
        is_forward : bool
            Is it a forward projection or backward projection. See below.

        -----------------------------------------------------------------------

        If is_forward == True, then forward projection:
        Projects from the INPUT of the start_layer to the OUTPUT of the
        end_layer:
                           start_layer                end_layer
                       -----------------          ----------------
                               |                         |
                         input |  output    ...    input |  output
                               |                         |
                       -----------------          ----------------     
        projection:        * --------------------------------->

        -----------------------------------------------------------------------

        If is_forward == False, then backward projection:
        Projects from the OUTPUT of the start_layer to the INPUT of the
        end_layer:
                            end_layer                start_layer
                       -----------------          ----------------
                               |                         |
                         input |  output    ...    input |  output
                               |                         |
                       -----------------          ----------------  
        projection:         <-------------------------------- *

        -----------------------------------------------------------------------

        This means that it is totally legal to have a start_layer_index that
        is equal to the end_layer_index. For example, the code:

            converter = SpatialIndexConverter(model, (227, 227))
            coord = converter.convert((111,111), 0, 0, is_forward=True)

        will return the coordinates of a box that include all output points
        of layer no.0 that can be influenced by the input pixel at (111,111).
        On the other hand, the code:

            coord = converter.convert((28,28), 0, 0, is_forward=False)

        will return the coordinates of a box that include all input pixels
        that can influence the output (of layer no.0) at (28,28).
        """
        vx, hx = self._process_index(index, start_layer_index)
        vx_min, vx_max = vx, vx
        hx_min, hx_max = hx, hx

        start_layer_name = self.idx_to_node[start_layer_index]
        end_layer_name = self.idx_to_node[end_layer_index]
        if is_forward:
            return self._forward_convert(vx_min, hx_min, vx_max, hx_max,
                                         start_layer_name, end_layer_name)
        else:
            return self._backward_convert(vx_min, hx_min, vx_max, hx_max, 
                                          start_layer_name, end_layer_name)
        # Return format: (vx_min, hx_min, vx_max, hx_max)


if __name__ == '__main__':
    model = models.resnet18()
    converter = SpatialIndexConverter(model, (227, 227))
    coord = converter.convert((64, 64), 21, 0, is_forward=False)
    print(coord)

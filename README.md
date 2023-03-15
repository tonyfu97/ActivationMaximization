# Deep Dream at the Neuron Level
UW CSE 455 Final Project

For details, please visit the [project website](https://github.com/tonyfu97/DeepDreamAtNeuronLevel)

File structure:
```shell
.
├── LICENSE
├── README.md
├── data
│   ├── model_info.txt
│   └── top_100_image_patches
│       ├── alexnet
│       ├── resnet18
│       └── vgg16
├── docs
│   ├── css
│   │   ├── gallery.css
│   │   ├── index.css
│   │   └── playground.css
│   ├── gallery.html
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── gallery.js
│   │   └── playground.js
│   ├── onnx_files
│   ├── playground.html
│   └── spike.m4a
├── python
│   ├── convert_to_onnx.py
│   ├── correlate_different_initializations.py
│   ├── grad_ascent.py
│   ├── grad_ascent_animation.py
│   ├── image_utils.py
│   ├── make_plots.py
│   ├── make_top_patch_initialized_grad_ascent.py
│   ├── make_top_patch_png.py
│   ├── make_zero_initialized_grad_ascent.py
│   ├── model_utils.py
│   ├── requirements.txt
│   ├── spatial_utils.py
│   └── tensor_utils.py
├── results
│   ├── SGD
│   │   ├── correlation
│   │   ├── top_patch_initialized
│   │   └── zero_initialized
│   ├── gif
│   │   └── alexnet
│   └── top_patch
│       └── alexnet
```

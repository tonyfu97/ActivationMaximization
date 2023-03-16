# DeepDream at the Neuron Level
UW CSE 455 Final Project

For details, please visit the [project website](https://tonyfu97.github.io/DeepDreamAtNeuronLevel/)



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
│   ├── index.html
│   ├── gallery.html
│   ├── playground.html
│   ├── css
│   │   ├── index.css
│   │   ├── gallery.css
│   │   └── playground.css
│   ├── js
│   │   ├── gallery.js
│   │   └── playground.js
│   ├── onnx_files
│   ├── images
│   └── spike.m4a
├── python
│   ├── requirements.txt
│   ├── convert_to_onnx.py
│   ├── grad_ascent_animation.py
│   ├── make_top_patch_png.py
│   ├── make_top_patch_initialized_grad_ascent.py
│   ├── make_zero_initialized_grad_ascent.py
│   ├── correlate_different_initializations.py
│   ├── make_plots.py
│   ├── grad_ascent.py
│   ├── image_utils.py
│   ├── model_utils.py
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

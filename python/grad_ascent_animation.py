"""
Play animation as the image is being applied gradient ascent. Will also save
the result as a GIF.

Tony Fu, Bair Lab, March 2023

"""

import os

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Custom modules
from model_utils import ModelInfo, get_truncated_model
from tensor_utils import process_tensor
from grad_ascent import GradientAscent

# Specify the model and layer
MODEL_NAME = 'alexnet'
LAYER_NAME = 'conv2'
UNIT_INDEX = 2

########################### DON'T TOUCH CODE BELOW ############################

# Setting up
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = getattr(models, MODEL_NAME)(pretrained=True).to(DEVICE)
MODEL_INFO = ModelInfo()

# Compute Gradient Ascent visualizations and save them to .png
NUM_ITER = 100
LR = 0.1

# Get layer specific information
img_size = MODEL_INFO.get_xn(MODEL_NAME, LAYER_NAME)
layer_index = MODEL_INFO.get_layer_index(MODEL_NAME, LAYER_NAME)
truncated_model = get_truncated_model(MODEL, layer_index)

# Initialize images
img1 = torch.zeros(1, 3, img_size, img_size, requires_grad=True, device=DEVICE)
img2 = torch.zeros(1, 3, img_size, img_size, requires_grad=True, device=DEVICE)
img3 = torch.zeros(1, 3, img_size, img_size, requires_grad=True, device=DEVICE)

# Create Gradient Ascent objects
ga1 = GradientAscent(truncated_model, UNIT_INDEX, img1, lr=LR, optimizer='SGD')
ga2 = GradientAscent(truncated_model, UNIT_INDEX, img2, lr=LR, optimizer='SGD', momentum=0.9)
ga3 = GradientAscent(truncated_model, UNIT_INDEX, img3, lr=LR, optimizer='Adam')

# Set up the figure for animation
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1, 3, 1)
ax1.axis("off")
ax1.set_title("SGD")
im1 = ax1.imshow(process_tensor(img1), vmin=0, vmax=1)
ax2 = fig.add_subplot(1, 3, 2)
ax2.axis("off")
ax2.set_title("SGD with momentum")
im2 = ax2.imshow(process_tensor(img2), vmin=0, vmax=1)
ax3 = fig.add_subplot(1, 3, 3)
ax3.axis("off")
ax3.set_title("ADAM")
im3 = ax3.imshow(process_tensor(img3), vmin=0, vmax=1)


def animate(i):
    global img1, img2, img3, ga1, ga2, ga3
    
    # Compute one gradient ascent
    ga1.step()
    ga2.step()
    ga3.step()
    
   # Update the displayed image
    im1.set_array(process_tensor(img1))
    im2.set_array(process_tensor(img2))
    im3.set_array(process_tensor(img3))

    fig.suptitle(f"Iteration {i+1}/{NUM_ITER}", fontsize=14)

    # Return the updated image
    return [im1, im2, im3]

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=NUM_ITER, blit=True)

# Save animation as GIF
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CURRENT_DIR, os.pardir, 'results', 'gif',
                           MODEL_NAME, LAYER_NAME)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

ani.save(os.path.join(RESULT_DIR, f"{UNIT_INDEX}_zero_initialized.gif"), writer='pillow')

# Show animation
plt.show()

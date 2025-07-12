import os
import torch
import numpy as np
from src.dataloader import VideoDataset
from src.transducers import get_transducer
from src.utils import quaternion_to_matrix
import matplotlib.pyplot as plt

# Define paths and device
# ------------------------
data_path = 'data/heart'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
# -------------
dataset = VideoDataset(data_path, 
                        device=device,
                        subsample_factor=None,
                        image_dtype=torch.float32,
                        )

# Initialize transducer
# ----------------------
if dataset.transducer_object is not None:
    transducer = dataset.transducer_object.to(device)
    print("Using pre-loaded transducer object.")
else:
    transducer = get_transducer(**dataset.transducer_params, device=device)
    print("Using transducer parameters to create transducer object.")

# Enable caching and load images
# ------------------------------
dataset.enable_cache()
images = dataset.image_cache.to(device)

# Extract poses and orientations
# ------------------------------
poses = dataset.poses.to(device)
positions = poses[:, :3]  # Nx3 positions
orientations = quaternion_to_matrix(poses[:, 3:7])  # Nx3x3 rotation matrices

# Visualize example images
# -------------------------
print("Visualizing example images from the dataset...")
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[np.random.randint(0, len(images))].cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Convert images to polar coordinates (if curvilinear transducer)
# --------------------------------------------------------------
if transducer.type == 'curvilinear':
    print("Converting images to polar coordinates...")
    images_converted = transducer.convert_scan_to_polar_coordinates(images.transpose(-2, -1).contiguous())

    # Visualize converted images
    print("Visualizing images in polar coordinates...")
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images_converted[np.random.randint(0, len(images_converted))].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    images_converted = images.transpose(-2, -1).contiguous()

# Visualize sample points
# ------------------------
sample_points_transducer = transducer.sample_points
plt.figure(figsize=(10, 5))

# Sample points in transducer plane
plt.subplot(1, 2, 1)
plt.scatter(sample_points_transducer[:, :, 0].cpu().numpy(), sample_points_transducer[:, :, 1].cpu().numpy(), s=1, label='Sample Points')
plt.xlabel('X coordinates [m]')
plt.ylabel('Y coordinates [m]')
plt.title('Sample Points in Transducer Plane')
plt.axis('equal')
plt.legend()

# Example converted image
plt.subplot(1, 2, 2)
plt.title('Example Converted Image')
plt.imshow(images_converted[0].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.show()


# Compute sample points in global coordinates
# -------------------------------------------
batch_size = 4
print("Computing sample points in global coordinates...")
sample_points_global = transducer.get_transducer_sample_points_global(positions=positions[:batch_size],
                                                                       orientations=orientations[:batch_size])


# Points: Batch x Height x Width x 3 (XYZ)
print("Sample points in global coordinates shape:", sample_points_global['points'].shape)

# Image: Batch x 1 x Height x Width
print("Corresponding image shape:", images_converted[:batch_size].shape)

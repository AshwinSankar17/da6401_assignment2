import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Path to the iNaturalist dataset root
dataset_path = '/projects/data/astteam/sparsh_assignment2/da6401_assignment2/dataset/inaturalist_12K/train'

# Define transform without normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Optional: resize for uniformity
    transforms.CenterCrop(224),     # Optional: crop to a fixed size
    transforms.ToTensor()           # Converts image to [0,1] and to CHW format
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Compute mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
n_samples = 0

print("Computing mean and std...")

for images, _ in tqdm(loader):
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)  # Flatten H and W
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std Deviation: {std}")

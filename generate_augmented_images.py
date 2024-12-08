import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Define base directory for data
BASE_DIR = "./data"
CATEGORIES = [
    "original",
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
    "mirror_upsidedown",
    "upsidedown_mirror",
    "invert_colors"
]

# Ensure output directories exist
OUTPUT_DIRS = {category: os.path.join(BASE_DIR, category) for category in CATEGORIES}
for dir_path in OUTPUT_DIRS.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Define transformations
# These define how images are modified for augmentation
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # Flip horizontally
vertical_flip = transforms.RandomVerticalFlip(p=1.0)      # Flip vertically
rotate_180 = transforms.RandomRotation((180, 180))        # Rotate the image 180 degrees
invert_colors = transforms.Lambda(lambda x: 1 - x)        # Invert pixel colors (white to black, black to white)
mirror_upsidedown = transforms.Compose([horizontal_flip, rotate_180])  # Mirror and then flip upside down
upsidedown_mirror = transforms.Compose([rotate_180, horizontal_flip])  # Flip upside down, then mirror

# Load the original MNIST dataset
original_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

# Augmentation logic
class AugmentedMNIST(Dataset):
    """
    Dataset wrapper to apply augmentations to each image in the MNIST dataset.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.augmentations = {
            "original": None,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "rotate_180": rotate_180,
            "mirror_upsidedown": mirror_upsidedown,
            "upsidedown_mirror": upsidedown_mirror,
            "invert_colors": invert_colors
        }

    def __len__(self):
        # Length is the number of original images multiplied by augmentation categories
        return len(self.base_dataset) * len(self.augmentations)

    def __getitem__(self, index):
        # Determine which image and augmentation to apply
        base_index = index // len(self.augmentations)  # Index of the original image
        aug_key = list(self.augmentations.keys())[index % len(self.augmentations)]  # Augmentation category

        # Get the original image and label
        img, label = self.base_dataset[base_index]

        # Apply the selected augmentation, if any
        if self.augmentations[aug_key]:
            img = self.augmentations[aug_key](img)

        return img, label, aug_key  # Return augmented image, label, and augmentation category

# Save augmented images
def save_augmented_images():
    """
    Generate and save augmented images in their respective category directories.
    """
    dataset = AugmentedMNIST(original_data)

    for img, label, aug_key in dataset:
        output_dir = OUTPUT_DIRS[aug_key]
        img_path = os.path.join(output_dir, f"{label}_{hash(img.numpy().tobytes())}.png")
        plt.imsave(img_path, img.squeeze().numpy(), cmap="gray")

if __name__ == "__main__":
    save_augmented_images()
    print("Augmented images have been saved successfully.")

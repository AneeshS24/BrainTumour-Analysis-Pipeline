import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.pyplot as plt

class BrainTumorClassificationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, noise_prob=0.05):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.label_dir = label_dir
        self.transform = transform
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        image = Image.open(img_path).convert("RGB")
        label = 1 if os.path.exists(label_path) and os.path.getsize(label_path) > 0 else 0
        if random.random() < self.noise_prob:
            label = 1 - label
        if self.transform:
            image = self.transform(image)
        return image, label

def visualize_samples(dataset, num_samples=6):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        image, label = dataset[i]
        image_np = image.permute(1, 2, 0).numpy()
        plt.subplot(1, num_samples, i+1)
        plt.imshow(image_np)
        plt.title("Tumor" if label == 1 else "No Tumor")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BrainTumorClassificationDataset(
        image_dir="archive/BrainTumor/BrainTumorYolov9/train/images",
        label_dir="archive/BrainTumor/BrainTumorYolov9/train/labels",
        transform=transform,
        noise_prob=0.05
    )

    print(f"Total images found: {len(dataset)}")

    if len(dataset) == 0:
        print("No images found. Check the path and .jpg files.")
    else:
        print("Dataset loaded successfully. Visualizing samples...")
        visualize_samples(dataset)

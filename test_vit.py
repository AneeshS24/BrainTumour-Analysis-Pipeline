import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from dataset import BrainTumorClassificationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = BrainTumorClassificationDataset(
    image_dir=os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "train", "images"),
    label_dir=os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "train", "labels"),
    transform=transform
)

val_dataset = BrainTumorClassificationDataset(
    image_dir=os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "valid", "images"),
    label_dir=os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "valid", "labels"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")

torch.save(model.state_dict(), "vit_brain_tumor.pth")
print("Model saved as vit_brain_tumor.pth")

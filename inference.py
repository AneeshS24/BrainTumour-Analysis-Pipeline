import os
import csv
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained ViT model (relative path)
model_path = os.path.join("vit_brain_tumor.pth")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformations (should match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Directory containing test images (relative path)
test_dir = os.path.abspath(
    os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "test", "images")
)
image_paths = [
    os.path.join(test_dir, fname)
    for fname in os.listdir(test_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Create output directories
output_vis_dir = "output_visuals"
os.makedirs(output_vis_dir, exist_ok=True)

# Run inference
results = []
for path in image_paths:
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).logits
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_class = predicted_class.item()

    label = "Tumor" if predicted_class == 1 else "No Tumor"
    filename = os.path.basename(path)
    print(f"{filename} --> {label} (Confidence: {confidence:.4f})")

    results.append([filename, label, f"{confidence:.4f}"])

    # Save visualization
    plt.imshow(image)
    plt.title(f"{label} ({confidence:.2f})")
    plt.axis('off')
    plt.savefig(os.path.join(output_vis_dir, f"{filename}_pred.png"))
    plt.close()

# Save results to CSV
csv_path = "vit_predictions.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Prediction", "Confidence"])
    writer.writerows(results)

print(f"\nInference complete. Results saved to: {csv_path}")
print(f"Visual outputs saved in directory: {output_vis_dir}")

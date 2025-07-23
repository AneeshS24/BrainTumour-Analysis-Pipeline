import os
import pandas as pd
import random

def generate_risk_labels(labels_dir, output_csv):
    records = []

    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            image_name = file.replace(".txt", ".jpg")
            txt_path = os.path.join(labels_dir, file)

            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    area = width * height

                    # Add Gaussian noise to area
                    noise = random.gauss(0, 0.002)
                    area_noisy = max(0.0, area + noise)

                    # More nuanced risk thresholds
                    if area_noisy < 0.012:
                        risk = "Low"
                    elif area_noisy < 0.028:
                        risk = "Medium"
                    else:
                        risk = "High"

                    records.append({
                        "image": image_name,
                        "class_id": class_id,
                        "area": area_noisy,
                        "risk": risk
                    })

    if not records:
        print("No valid labels found. Check label directory path and file content.")
    else:
        df = pd.DataFrame(records)
        df.to_csv(output_csv, index=False)
        print(f"Risk-labeled data saved to: {output_csv}")

if __name__ == "__main__":
    label_folder = os.path.join("runs", "detect", "predict_brain", "labels")
    output_csv = os.path.join("risk_labeled_data.csv")
    generate_risk_labels(label_folder, output_csv)

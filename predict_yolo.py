from ultralytics import YOLO
import os
import pandas as pd

def run_detection():
    # Load trained YOLOv8 model
    model = YOLO("runs/detect/brain_tumor_yolov8/weights/best.pt")

    # Relative path to test images
    test_dir = os.path.join("archive", "BrainTumor", "BrainTumorYolov9", "test", "images")

    # Run object detection
    results = model.predict(
        source=test_dir,
        conf=0.5,
        save=True,
        save_txt=True,
        project="runs/detect",
        name="predict_brain",
        exist_ok=True
    )

    print("Detection complete.")
    print("Results saved to: runs/detect/predict_brain")

    # Paths for saved results
    save_dir = os.path.join("runs", "detect", "predict_brain")
    labels_dir = os.path.join(save_dir, "labels")

    predictions = []

    # Iterate through label files
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith(".txt"):
            image_name = txt_file.replace(".txt", ".jpg")
            txt_path = os.path.join(labels_dir, txt_file)

            with open(txt_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        cls_id, x_center, y_center, width, height, conf = parts
                        predictions.append({
                            "image": image_name,
                            "class_id": int(cls_id),
                            "confidence": float(conf),
                            "x_center": float(x_center),
                            "y_center": float(y_center),
                            "width": float(width),
                            "height": float(height)
                        })

    # Save predictions to CSV
    df = pd.DataFrame(predictions)
    csv_path = os.path.join(save_dir, "yolo_predictions.csv")
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved to: {csv_path}")

if __name__ == "__main__":
    run_detection()

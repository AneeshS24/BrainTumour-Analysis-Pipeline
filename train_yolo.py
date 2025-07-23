from ultralytics import YOLO

def train_yolo():
    # Load YOLOv8 model (YOLOv8s is lightweight)
    model = YOLO('yolov8s.pt')  # Can change to yolov8n.pt (nano) or yolov8m.pt (medium)

    # Train model
    model.train(
        data='archive/BrainTumor/BrainTumorYolov9/data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        name='brain_tumor_yolov8',
        device=0  # Use 0 for GPU, 'cpu' if no GPU
    )

if __name__ == "__main__":
    train_yolo()

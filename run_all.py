import os
import subprocess

# Paths to all stages
classification_script = os.path.join("classification", "train_vit.py")
predict_script = "inference.py"
risk_label_script = os.path.join("risk_classification", "generate_risk_labels.py")
risk_classifier_script = os.path.join("risk_classification", "train_risk_classifier.py")
survival_script = os.path.join("survival_estimation", "simulate_survival.py")

print("Step 1: Training Vision Transformer on classification...")
subprocess.run(["python", classification_script])

print("\nStep 2: Running ViT inference...")
subprocess.run(["python", predict_script])

print("\nStep 3: Generating risk labels based on YOLO detections...")
subprocess.run(["python", risk_label_script])

print("\nStep 4: Training risk classifier...")
subprocess.run(["python", risk_classifier_script])

print("\nStep 5: Simulating survival estimation...")
subprocess.run(["python", survival_script])

print("\nPipeline completed! Check generated CSVs and model weights.")

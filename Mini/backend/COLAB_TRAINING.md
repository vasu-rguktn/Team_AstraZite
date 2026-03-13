
# Microplastic Detection - Colab Training Guide

This guide contains the optimized code for training your microplastic detection model on Google Colab.

## 1. Setup Environment
Open a new notebook in Google Colab, go to **Runtime > Change runtime type**, and select **GPU (T4 or better)**.

```python
# Install YOLOv8 and Roboflow libraries
!pip install ultralytics roboflow
```

## 2. Load Dataset
Replace `YOUR_API_KEY` with your Roboflow API key (found in your Roboflow settings).

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("iam").project("microplastics-m7mf5")
version = project.version(1)
dataset = version.download("yolov8")
```

## 3. High-Accuracy Training
Using YOLOv8 Small (`yolov8s.pt`) is recommended for 400 images. It captures micro-details better than the Nano version without overfitting as easily as larger versions.

```python
from ultralytics import YOLO

# Load pre-trained weights
model = YOLO('yolov8s.pt')

# Optimized training for small datasets
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,            # High epoch limit with early stopping
    patience=30,           # Stop if no improvement for 30 epochs (anti-overfitting)
    imgsz=640,             # Keep high resolution for small particles
    batch=16,              
    optimizer='AdamW',      # Advanced optimizer for better convergence
    lr0=0.001,             
    augment=True,          # Essential for small datasets (mosaic, blur, etc.)
    box=7.5,               # Increase focus on box accuracy
    cls=0.5,               # Balance class loss
    device=0               # Use GPU
)
```

## 4. Download Trained Weights
After training, your best model will be saved at `runs/detect/train/weights/best.pt`.

```python
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')
```

---

## 5. Local Setup (VS Code)
1. Download the `best.pt` file.
2. In your local project, place it inside the `model/` folder.
3. Run `python main.py` to see the results.

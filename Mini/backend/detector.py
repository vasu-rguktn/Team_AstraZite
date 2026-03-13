from ultralytics import YOLO
import cv2
import numpy as np
import config
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class MicroplasticDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, image, conf=config.CONFIDENCE_THRESHOLD):
        """
        Run inference on an image.
        Returns the results object from Ultralytics.
        """
        if self.model is None:
            return None
        
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=config.IOU_THRESHOLD,
            save=False,
            verbose=False
        )
        return results[0]

    def get_detections(self, result):
        """
        Parses results into a clean list of dictionaries.
        """
        detections = []
        if result is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else "Unknown"
            
            detections.append({
                'bbox': [int(c) for c in coords],
                'confidence': conf,
                'class_id': cls_id,
                'name': name
            })
        
        return detections

class AIVsRealClassifier:
    def __init__(self, model_path=config.AI_VS_REAL_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            # Initialize model architecture (ResNet18 as used in training)
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            
            # Load weights
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"AI vs Real model loaded successfully from {model_path}")
            else:
                print(f"Warning: AI vs Real model not found at {model_path}. Using random weights.")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading AI vs Real model: {e}")
            self.model = None

    def predict(self, image_cv2):
        """
        Takes CV2 image (BGR), converts to RGB PIL, and predicts Real vs AI.
        """
        if self.model is None:
            return "Unknown", "0.0%"
            
        try:
            # Convert OpenCV (BGR) to PIL (RGB)
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Transform
            img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                conf, predicted = torch.max(probabilities, 1)
                
            class_idx = predicted.item()
            confidence_val = conf.item() * 100
            
            # Labels: 0 = Synthetic/AI, 1 = Real
            label = "Invalid / Synthetic" if class_idx == 0 else "Real Sample"
            confidence_str = f"{confidence_val:.1f}%"
            
            return label, confidence_str
        except Exception as e:
            print(f"Prediction error in AI vs Real: {e}")
            return "Error", "0.0%"

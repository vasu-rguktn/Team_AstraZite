
import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from detector import MicroplasticDetector, AIVsRealClassifier
from utils import draw_detections, create_crop_montage
import config

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detectors
detector = MicroplasticDetector()
real_vs_ai = AIVsRealClassifier()

def get_contamination_level(count):
    if count == 0:
        return "None"
    elif count < 5:
        return "Low"
    elif count < 15:
        return "Moderate"
    elif count < 30:
        return "High"
    else:
        return "Extreme"

@app.post("/predict")
async def predict_microplastics(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"status": "error", "message": "Invalid image format"}

        # Run detection
        result = detector.predict(image)
        detections = detector.get_detections(result)
        
        # Run Real vs AI prediction
        image_type, image_type_confidence = real_vs_ai.predict(image)
        
        # Sample Quality Guard: If it's 100% not real, it's likely a non-microscopic image
        is_valid_sample = (image_type == "Real Sample")
        
        # Only run full detection if it's potentially a real sample
        # or show it but with a clear disclaimer
        
        # Strict Quality Guard
        is_valid_sample = (image_type == "Real Sample")
        
        if not is_valid_sample:
            # Block detections for non-microscopic images
            detections = []
            counts = {name: 0 for name in config.CLASS_NAMES}
        else:
            # Calculate counts for real samples
            counts = {name: 0 for name in config.CLASS_NAMES}
            for det in detections:
                name = det['name']
                if name in counts:
                    counts[name] += 1
                else:
                    counts[name] = counts.get(name, 0) + 1

        # Visualize results
        annotated_image = draw_detections(image, detections)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Determine contamination level
        total_count = len(detections)
        contamination_level = get_contamination_level(total_count)

        # Primary prediction logic
        if not is_valid_sample:
            primary_prediction = "INVALID SAMPLE"
            contamination_level = "None"
        elif total_count > 0:
            primary_prediction = max(counts, key=counts.get).upper()
        else:
            primary_prediction = "CLEAR"

        # Real vs AI information
        # image_type and image_type_confidence were calculated earlier
        
        return {
            "status": "success",
            "prediction": primary_prediction,
            "total_microplastics": total_count,
            "image": img_base64,
            "contamination_level": contamination_level,
            "counts": counts,
            "image_type": image_type,
            "image_type_confidence": image_type_confidence
        }

    except Exception as e:
        return {"status": "error", "message": f"Server error: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Microplastic Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

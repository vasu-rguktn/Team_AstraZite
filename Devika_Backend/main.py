from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classes (Alphabetical order as per ImageFolder training)
CLASS_NAMES = ["algae", "filament", "fragment", "pellet"]
DETAILS = {
    "pellet": "Primary microplastic used in manufacturing. Commonly found as small beads in water.",
    "filament": "Microfibers from fishing nets, synthetic clothes, and ropes.",
    "fragment": "Broken pieces of larger plastic waste, often irregular shapes.",
    "algae": "Natural organic material (non-plastic) present in water samples."
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 4)
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "backend", "model.pth")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform for Microplastic Detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# AI vs Real Classification Model
AI_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ai_vs_real.pth")
ai_vs_real_model = models.resnet18(weights=None)
ai_vs_real_model.fc = torch.nn.Linear(ai_vs_real_model.fc.in_features, 2)

if os.path.exists(AI_MODEL_PATH):
    ai_vs_real_model.load_state_dict(torch.load(AI_MODEL_PATH, map_location=DEVICE))
    ai_vs_real_model.loaded = True
    print("AI vs Real model loaded successfully.")
else:
    ai_vs_real_model.loaded = False
    print(f"Warning: AI vs Real model not found at {AI_MODEL_PATH}")

ai_vs_real_model.to(DEVICE)
ai_vs_real_model.eval()

# Transform for AI vs Real (Standard ImageNet normalization as used in training)
ai_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 0. AI vs Real Analysis
        is_ai_generated = False
        ai_conf_val = 0.0
        if getattr(ai_vs_real_model, 'loaded', False):
            ai_tensor = ai_transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ai_output = ai_vs_real_model(ai_tensor)
                ai_probs = F.softmax(ai_output, dim=1)
                ai_choice = torch.argmax(ai_probs, dim=1).item()
                is_ai_generated = (ai_choice == 1)
                ai_conf_val = ai_probs[0][ai_choice].item()

        # 1. Global Analysis 
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            global_probs = F.softmax(output, dim=1)
            global_conf, global_idx = torch.max(global_probs, 1)
            
            top2_probs, _ = torch.topk(global_probs, 2)
            margin = top2_probs[0][0] - top2_probs[0][1]

        label = CLASS_NAMES[global_idx.item()]
        
        # Rejection Logic for Unrelated Images (Humans, objects, etc.)
        # For AI Generated samples, we use more lenient thresholds as they are known to be relevant microscopic simulations
        rej_conf_thresh = 0.60 if not is_ai_generated else 0.35
        rej_margin_thresh = 0.12 if not is_ai_generated else 0.05

        if global_conf.item() < rej_conf_thresh or margin < rej_margin_thresh:
            return {
                "status": "unrelated",
                "message": "Image rejected as non-relevant content. Please upload a clear microscopic sample.",
                "prediction": "Unknown",
                "confidence": f"{global_conf.item() * 100:.2f}%",
                "details": "Image does not match microscopic profiles.",
                "image_type": "AI Generated" if is_ai_generated else "Real Microscopic",
                "image_type_confidence": f"{ai_conf_val * 100:.2f}%"
            }

        # 2. Multi-Scale Analysis (Capture both small and large particles)
        # We combine 3x3 grid and 4x4 grid for better coverage
        patch_tensors = []
        
        # 3x3 Grid
        for i in range(3):
            for j in range(3):
                w, h = image.size
                pw, ph = w // 3, h // 3
                box = (j * pw, i * ph, (j + 1) * pw, (i + 1) * ph)
                patch_tensors.append(transform(image.crop(box).resize((224, 224))))
        
        # 4x4 Grid
        for i in range(4):
            for j in range(4):
                w, h = image.size
                pw, ph = w // 4, h // 4
                box = (j * pw, i * ph, (j + 1) * pw, (i + 1) * ph)
                patch_tensors.append(transform(image.crop(box).resize((224, 224))))

        batch_tensor = torch.stack(patch_tensors).to(DEVICE)
        with torch.no_grad():
            patch_outputs = model(batch_tensor)
            patch_probs = F.softmax(patch_outputs, dim=1)
            patch_conf, patch_idx = torch.max(patch_probs, 1)

        # 3. Aggregate Results using a hybrid approach
        counts = {name: 0 for name in CLASS_NAMES}
        total_valid_patches = 0
        summed_probs = global_probs[0].clone() # Start with global context weight
        
        for i in range(len(patch_tensors)):
            p_probs = patch_probs[i]
            p_conf = patch_conf[i].item()
            summed_probs += p_probs # Accumulate for distribution analysis
            
            if p_conf > 0.45: # Balanced threshold for counting
                p_label = CLASS_NAMES[patch_idx[i].item()]
                counts[p_label] += 1
                total_valid_patches += 1

        # Use probability distribution for the most accurate percentage
        avg_probs = summed_probs / (len(patch_tensors) + 1)
        algae_pct = avg_probs[0].item() * 100 # Algae is index 0
        
        # Ensure if global is Algae, it's represented in counts even if patches are split
        if label == "algae" and counts["algae"] == 0:
            counts["algae"] = 1 # Force at least 1 count to match global prediction
            total_valid_patches = max(total_valid_patches, 1)

        mp_total = counts["pellet"] + counts["filament"] + counts["fragment"]
        
        formatted_counts = {
            "pellet": counts["pellet"],
            "filament": counts["filament"],
            "fragment": counts["fragment"],
            "algae I": counts["algae"]
        }

        return {
            "status": "success",
            "prediction": label, 
            "confidence": f"{global_conf.item() * 100:.2f}%",
            "details": DETAILS.get(label, "Detected aquatic content."),
            "total_microplastic": mp_total,
            "algae_percentage": f"{algae_pct:.1f}%", # This is the "Purity" percentage
            "counts": formatted_counts,
            "total_particles": total_valid_patches,
            "detected_objects": len(patch_tensors),
            "image_type": "AI Generated" if is_ai_generated else "Real Microscopic",
            "image_type_confidence": f"{ai_conf_val * 100:.2f}%"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

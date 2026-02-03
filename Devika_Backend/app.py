import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Classes
CLASS_NAMES = ["pellet", "filament", "fragment", "algae"]
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
model.load_state_dict(torch.load("backend/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit UI
st.set_page_config(page_title="🌊 Microplastic Detection", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1E90FF;'>🌊 Microplastic Detection in Water Samples</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload an image of water sample", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    label = CLASS_NAMES[pred_idx.item()]
    conf = confidence.item() * 100

    # Attractive information layout
    st.markdown("### Detection Results")
    col1, col2 = st.columns([1,2])
    with col1:
        if label == "algae":
            st.success("🟢 Non-Plastic Detected")
        else:
            st.error("🔴 Plastic Detected")
        st.metric("Confidence", f"{conf:.2f}%")
        st.metric("Type", label.capitalize())
    with col2:
        st.markdown(f"**Details:** {DETAILS[label]}")

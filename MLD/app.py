import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(
    page_title="Microplastic Detection System",
    page_icon="🌊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("microplastic_multiclass_model.h5")

model = load_model()
class_names = ['algae', 'filament', 'fragment', 'pellet']

st.title("🌊 Microplastic Detection & Classification")
st.caption("Detect microplastic presence and identify its type")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a microscopic image (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        img = img.resize((224, 224))
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr)[0]

        # Get top two probabilities
        sorted_idx = np.argsort(preds)
        idx1 = sorted_idx[-1]
        idx2 = sorted_idx[-2]

        p1 = preds[idx1] * 100
        p2 = preds[idx2] * 100
        label = class_names[idx1]

        st.subheader("📊 Model Output")
        for i, cls in enumerate(class_names):
            st.write(f"{cls.capitalize()}: {preds[i]*100:.2f}%")

        st.markdown("---")

        # -------- FINAL LOGIC --------
        if (p1 - p2) >= 5:
            st.success("✅ MICROPLASTIC DETECTED")
            st.success(f"🧪 Type: **{label.upper()}**")
            st.write(f"Dominance Margin: {p1 - p2:.2f}%")
            st.progress(int(p1))
        else:
            st.error("❌ NON-MICROPLASTIC / BACKGROUND IMAGE")
            st.write("No dominant microplastic pattern detected.")

st.markdown("---")
st.caption(
    "Note: Detection is based on dominance analysis of CNN outputs on microscopic images."
)

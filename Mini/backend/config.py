
# Configuration for Microplastic Detection Project

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
AI_VS_REAL_MODEL_PATH = os.path.join(BASE_DIR, "ai_vs_real.pth")
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class names based on Roboflow dataset
CLASS_NAMES = ['fiber', 'film', 'fragment', 'pallet']

# Visualization settings
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
COLORS = {
    'fiber': (0, 0, 255),    # Red
    'film': (255, 255, 0),   # Cyan/Yellow-ish
    'fragment': (255, 0, 255), # Magenta
    'pallet': (0, 255, 0)     # Green
}

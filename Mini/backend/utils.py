
import cv2
import config
import numpy as np

def draw_detections(image, detections):
    """
    Draws stylized bounding boxes and labels on the image.
    Matches the 'high accuracy / premium' look requested.
    """
    annotated_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        name = det['name']
        conf = det['confidence']
        
        # Get color for class
        color = config.COLORS.get(name, (255, 255, 255))
        
        # Draw bounding box with rounded corners (simulated)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)
        
        # Label text
        label = f"{name} {conf:.2f}"
        
        # Get text size
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, 1)
        
        # Draw background for label
        cv2.rectangle(annotated_image, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        
        # Draw text
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, config.TEXT_SCALE, (0, 0, 0), 1)

    return annotated_image

def create_crop_montage(image, detections, max_crops=4):
    """
    Optional: Create a montage of cropped particles for closer inspection.
    Useful for high-accuracy verification.
    """
    if not detections:
        return None
        
    crops = []
    for det in detections[:max_crops]:
        x1, y1, x2, y2 = det['bbox']
        # Add some padding to crop
        h, w, _ = image.shape
        pad = 10
        x1_p = max(0, x1 - pad)
        y1_p = max(0, y1 - pad)
        x2_p = min(w, x2 + pad)
        y2_p = min(h, y2 + pad)
        
        crop = image[y1_p:y2_p, x1_p:x2_p]
        if crop.size > 0:
            crop = cv2.resize(crop, (150, 150))
            # Build border
            color = config.COLORS.get(det['name'], (255, 255, 255))
            crop = cv2.copyMakeBorder(crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)
            crops.append(crop)
            
    if crops:
        return np.hstack(crops)
    return None

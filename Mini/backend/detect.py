from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("model/best.pt")

# Read image
image_path = "test_images/sample.jpg"
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Draw bounding boxes
annotated_frame = results[0].plot()

# Show result
cv2.imshow("Microplastic Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
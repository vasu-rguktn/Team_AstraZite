
import cv2
import os
from detector import MicroplasticDetector
from utils import draw_detections, create_crop_montage
import config

def main():
    # 1. Initialize detector
    detector = MicroplasticDetector()
    
    # 2. Path to input image
    # Note: Ensure you have an image in test_images/ folder
    input_path = "test_images/sample.jpg" 
    
    if not os.path.exists(input_path):
        # Create folder if it doesn't exist
        os.makedirs("test_images", exist_ok=True)
        print(f"Error: {input_path} not found.")
        print("Please place your microscopic image at: test_images/sample.jpg")
        return

    # 3. Load image
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Could not decode image. Check path and file format.")
        return

    # 4. Run detection
    print("Detecting microplastics...")
    result = detector.predict(image)
    detections = detector.get_detections(result)

    # 5. Visualize
    output_image = draw_detections(image, detections)
    
    # 6. Show and Save results
    cv2.imshow("Detected Microplastics", output_image)
    cv2.imwrite("output_result.jpg", output_image)
    print("Main detection result saved to output_result.jpg")
    
    # 7. Zoomed Montage
    montage = create_crop_montage(image, detections)
    if montage is not None:
        cv2.imshow("Crops (Zoomed)", montage)
        cv2.imwrite("output_crops.jpg", montage)
        print("Crops montage saved to output_crops.jpg")

    print(f"--- Detection Summary ---")
    print(f"Total particles detected: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  {i+1}: {d['name']} ({d['confidence']:.2f})")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2

# Load detection model (pretrained or your custom trained one)
model = YOLO("yolov8n.pt")  # or your trained cattle detector

def detect_cattle(image_path):
    results = model(image_path)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None

    # Take first detected cattle
    x1, y1, x2, y2 = map(int, boxes[0])

    image = cv2.imread(image_path)
    crop = image[y1:y2, x1:x2]

    crop_path = image_path.replace(".jpg", "_crop.jpg")
    cv2.imwrite(crop_path, crop)

    return crop_path

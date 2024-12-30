import torch
from ultralytics import YOLO
import pytesseract
from PIL import Image
import cv2
import numpy as np

def initialize_model(model_path="yolov5s.pt"):
    """
    Initialize the YOLO model for object detection.

    Args:
        model_path (str): Path to the YOLO model.

    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_path)
    return model

def preprocess_image(image_path):
    """
    Load the image and convert it to OpenCV format.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Loaded image in OpenCV format.
    """
    image = Image.open(image_path).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def perform_detection(model, image):
    """
    Perform object detection on the image using YOLO.

    Args:
        model (YOLO): YOLO object detection model.
        image (np.ndarray): Input image in OpenCV format.

    Returns:
        list: Detection results containing bounding boxes, confidences, and class IDs.
    """
    results = model(image)
    return results[0].boxes.data.cpu().numpy()

def draw_bounding_boxes(image, detections, threshold=0.25):
    """
    Draw bounding boxes on the image for detected objects.

    Args:
        image (np.ndarray): Original image in OpenCV format.
        detections (list): Detection results from YOLO.
        threshold (float): Confidence threshold for displaying detections.

    Returns:
        np.ndarray: Image with bounding boxes drawn.
    """
    for detection in detections:
        x0, y0, x1, y1, confidence, class_id = detection
        if confidence > threshold:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(image, f"Conf: {confidence:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def perform_ocr_on_box(image, box):
    """
    Perform OCR on the specified bounding box region of the image.

    Args:
        image (np.ndarray): Image in OpenCV format.
        box (tuple): Bounding box coordinates (x0, y0, x1, y1).

    Returns:
        str: Detected text from the bounding box region.
    """
    x0, y0, x1, y1 = map(int, box)
    cropped_image = image[y0:y1, x0:x1]
    try:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(gray, config=config).strip()
    except Exception as e:
        text = f"Error: {e}"
    return text

def main(image_path, model_path="yolov5su.pt"):
    """
    Main function to process an image, detect objects, and perform OCR.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLO model.
    """
    model = initialize_model(model_path)
    image = preprocess_image(image_path)
    detections = perform_detection(model, image)
    image_with_boxes = draw_bounding_boxes(image, detections)

    for detection in detections:
        x0, y0, x1, y1, confidence, class_id = detection
        if confidence > 0.25:  # Ensure only confident detections are processed
            number_plate_text = perform_ocr_on_box(image, (x0, y0, x1, y1))
            cv2.putText(image_with_boxes, f"Plate: {number_plate_text}", (int(x0), int(y0) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow(image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "images/vtest.jpg"
    main(image_path)

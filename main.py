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

def preprocess_frame(frame):
    """
Convert a single video frame to RGB format for YOLO processing.

Args:
frame (np.ndarray): Input frame in OpenCV format.

Returns:
np.ndarray: Frame in RGB format.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def perform_detection(model, frame):
    """
Perform object detection on a single video frame using YOLO.

Args:
model (YOLO): YOLO object detection model.
frame (np.ndarray): Input frame in RGB format.

Returns:
list: Detection results containing bounding boxes, confidences, and class IDs.
    """
    results = model(frame)
    return results[0].boxes.data.cpu().numpy()

def draw_bounding_boxes(frame, detections, threshold=0.25):
    """
Draw bounding boxes on a video frame for detected objects.

Args:
frame (np.ndarray): Original frame in OpenCV format.
detections (list): Detection results from YOLO.
threshold (float): Confidence threshold for displaying detections.

Returns:
np.ndarray: Frame with bounding boxes drawn.
    """
    for detection in detections:
        x0, y0, x1, y1, confidence, class_id = detection
        if confidence > threshold:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def perform_ocr_on_box(frame, box):
    """
Perform OCR on the specified bounding box region of a video frame.

Args:
frame (np.ndarray): Frame in OpenCV format.
box (tuple): Bounding box coordinates (x0, y0, x1, y1).

Returns:
str: Detected text from the bounding box region.
    """
    x0, y0, x1, y1 = map(int, box)
    cropped_image = frame[y0:y1, x0:x1]
    try:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(gray, config=config).strip()
    except Exception as e:
        text = f"Error: {e}"
    return text

def main_realtime(model_path="yolov5s.pt"):
    """
Main function to process video frames in real-time, detect objects, and perform OCR.

Args:
model_path (str): Path to the YOLO model.
    """
    model = initialize_model(model_path)
    cap = cv2.VideoCapture(0)  # Open default webcam

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        rgb_frame = preprocess_frame(frame)
        detections = perform_detection(model, rgb_frame)
        frame_with_boxes = draw_bounding_boxes(frame, detections)

        for detection in detections:
            x0, y0, x1, y1, confidence, class_id = detection
            if confidence > 0.25:  # Ensure only confident detections are processed
                number_plate_text = perform_ocr_on_box(frame, (x0, y0, x1, y1))
                cv2.putText(frame_with_boxes, f"Plate: {number_plate_text}", (int(x0), int(y0) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('ANPR Realtime', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_realtime()

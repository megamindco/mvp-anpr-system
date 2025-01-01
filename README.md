# Automatic Number Plate Recognition (ANPR) System
This project implements an Automatic Number Plate Recognition (ANPR) system using YOLO for object detection and Tesseract for Optical Character Recognition (OCR). The system processes video frames in real-time, detects number plates, and performs OCR to extract the text from the detected plates.

# [Colab Notebook](https://colab.research.google.com/drive/1VO7NmE9qhktgB87cljYgf9acsVZcDE5Q?usp=sharing)
## Features
- Real-time object detection using YOLO.
- Optical Character Recognition (OCR) using Tesseract.
- Bounding box drawing for detected objects.
- Integration with camera for real-time processing.

## Requirements
- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLO
- Pytesseract
- PIL (Python Imaging Library)
- NumPy

## Installation
- ### Clone the repository:
```Bash
git clone https://github.com/megamindco/mvp-anpr-system.git
cd anpr-system
```

- ### Install the required dependencies:
```Bash
 python -m venv anpr
 venv\Scripts\activate
 pip install -r deps.txt
```

- ### Download the YOLO model weights (e.g., yolov5s.pt) and place them in the project directory.
- ### Download Tessrect-ocr for your platform

## Usage
Run the main script to start the real-time ANPR system:

```Bash
python main.py
```
The system will open the default cam and start processing video frames. Press q to exit the application.

## Code Overview
- ### Initialize Model
```Python
def initialize_model(model_path="yolov5s.pt"):
    model = YOLO(model_path)
    return model
```
- ### Preprocess Frame
```Python
def preprocess_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

- ### Perform Detection
```Python
def perform_detection(model, frame):
    results = model(frame)
    return results[0].boxes.data.cpu().numpy()
```
- ### Draw Bounding Boxes
```Python
def draw_bounding_boxes(frame, detections, threshold=0.25):
    for detection in detections:
        x0, y0, x1, y1, confidence, class_id = detection
        if confidence > threshold:
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
```
- ### Perform OCR on Box
```Python
def perform_ocr_on_box(frame, box):
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
```
- ### Main Real-time Processing

```Python

def main_realtime(model_path="yolov5s.pt"):
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

```
## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

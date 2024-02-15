import cv2
import pyttsx3
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the confidence threshold for detection
conf_threshold = 0.5

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Open the default camera
cap = cv2.VideoCapture(0)

# Continuously read frames from the camera and perform object detection
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame from camera.")
        break

    # Run the object detection model on the frame
    results = model(frame)

    # Get the detected object classes and their confidence scores
    detected_classes = results.pred[0][:, -1].cpu().numpy()
    conf_scores = results.pred[0][:, 4].cpu().numpy()


    # Visualize the detection results on the frame
    for i, (class_idx, conf) in enumerate(zip(detected_classes, conf_scores)):
        if conf > conf_threshold:
            label = model.model.names[int(class_idx)]
            cv2.putText(frame, label, (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            box = results.pred[0][i, :4].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Speak the name of the detected object
            engine.say(label)
            engine.runAndWait()

    # Display the frame with detection results
    cv2.imshow('Object Detection', frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources used
cap.release()
cv2.destroyAllWindows()

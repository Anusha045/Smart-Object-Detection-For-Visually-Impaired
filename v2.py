import numpy as np
import time
import cv2
import os
import imutils
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed
engine.setProperty('volume', 1.0)  # Adjust volume

# Load the COCO class labels YOLO was trained on
LABELS = open("coco.names").read().strip().split("\n")

# Read class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
font = cv2.FONT_HERSHEY_PLAIN

# Get layer names
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize camera
cap = cv2.VideoCapture(0)

frame_count = 0
start = time.time()

while True:
    frame_count += 1
    ret, frame = cap.read()
    cv2.imshow("Live Object Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if ret:
        if frame_count % 60 == 0:
            # Get frame dimensions
            (H, W) = frame.shape[:2]

            # Convert frame to blob for YOLO
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # Initialize lists for detection results
            boxes, confidences, classIDs, centers = [], [], [], []

            # Process YOLO detections
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            # Apply non-maxima suppression
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            detected_objects = ["Detected objects:"]

            # Process detected objects
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[classIDs[i]])
                    confidence = confidences[i]
                    color = colors[classIDs[i]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

                    # Determine object position
                    centerX, centerY = centers[i][0], centers[i][1]

                    W_pos = "left" if centerX <= W / 3 else "center" if centerX <= (W / 3 * 2) else "right"
                    H_pos = "top" if centerY <= H / 3 else "mid" if centerY <= (H / 3 * 2) else "bottom"

                    detected_objects.append(f"{H_pos} {W_pos} {LABELS[classIDs[i]]}")

            # Print detected objects
            detected_text = ', '.join(detected_objects)
            print(detected_text)

            # Read out the detected objects
            engine.say(detected_text)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()

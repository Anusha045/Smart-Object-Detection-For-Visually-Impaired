**Object Detection and Voice Guidance System for the Visually Impaired**

**Project Overview**

The Object Detection and Voice Guidance System is a smart, AI-powered assistive tool designed to enhance the situational awareness of visually impaired users. Leveraging real-time computer vision and audio feedback, the system detects nearby objects, estimates their distance, and provides spoken alerts — allowing the user to make safe navigation decisions, especially in dynamic environments like roads or public spaces.

**Real-Time Object Detection:**
Detects all objects in the camera’s view using the YOLOv3 deep learning model.

Distance Estimation: Approximates how far each object is from the user and prioritizes nearby threats.

Voice Feedback: Provides spoken alerts such as:

“Car approaching in 1 meter — move right.”

“Person detected on the left — 2 meters away.”

Motion Analysis: Identifies moving objects (e.g., vehicles) and gives directional guidance (left/right).

Multi-Object Awareness: Describes multiple objects with distance and position in real time.

Portable Setup: Works on standard laptops or embedded systems with a webcam.

** System Architecture**

Camera/Webcam: Captures live video feed.

YOLOv3 Model: Detects and classifies objects.

Distance Estimator: Calculates proximity based on object size and frame positioning.

Direction Identifier: Locates object position (left, center, right).

Text-to-Speech Engine (pyttsx3): Converts object and distance data into audible voice instructions.

Main Logic (v2.py): Core system logic that ties detection, direction, voice feedback, and motion alerting.


 **Technologies Used**
 
Programming Language: Python

Libraries: OpenCV, NumPy, pyttsx3, YOLOv3 (Darknet), imutils

Model: Pre-trained YOLOv3 for object detection

Audio Engine: pyttsx3 (offline TTS)

**Usage**

Start Camera: Application opens webcam and begins scanning.

Voice Alerts: Spoken feedback announces object type, distance, and movement.

Real-Time Decision Making: System provides left/right movement suggestions for safety.

**Future Enhancements**

GPS integration for outdoor navigation.

Dynamic learning model for personalized object detection.

Support for multi-camera environment.

Integration with smart wearable devices.

Multilingual voice feedback.



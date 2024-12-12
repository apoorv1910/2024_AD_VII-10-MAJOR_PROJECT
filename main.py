# Importing necessary libraries 
import cv2
import numpy as np
import jetson.inference
import jetson.utils
import threading

# Set up the DroidCam feed URL
# phone_camera_url = "http://192.168.29.76:4747/video"
# camera = cv2.VideoCapture(phone_camera_url)

camera = cv2.VideoCapture(1)

# Set camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS to ease CPU usage if needed

# Model selection
print("Select the model for object detection:")
print("1: SSD-Mobilenet-v2 (general object detection)")
print("2: PedNet (pedestrian detection)")
print("3: SSD-Inception-v2 (general object detection, heavier)")
print("4: MultiBox (face detection)")
choice = input("Enter the model number (1, 2, 3, or 4): ")

# Initialize model based on user choice
if choice == '1':
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)
elif choice == '2':
    net = jetson.inference.detectNet("pednet", threshold=0.6)
elif choice == '3':
    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.6)
elif choice == '4':
    net = jetson.inference.detectNet("multiped", threshold=0.6)
else:
    print("Invalid choice. Defaulting to SSD-Mobilenet-v2.")
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)

frame = None
frame_counter = 0
frame_skip = 2  # Skip every 2nd frame to reduce lag

# Define function to continuously capture frames in a separate thread
def capture_frames():
    global frame
    while True:
        ret, new_frame = camera.read()
        if ret:
            frame = new_frame

# Start the frame capture thread
thread = threading.Thread(target=capture_frames)
thread.start()

# Main loop for processing and displaying the video feed
while True:
    if frame is None:
        continue

    frame_counter += 1
    if frame_counter % frame_skip != 0:  # Skip frames to reduce lag
        continue

    # Convert to CUDA format for Jetson processing
    img_cuda = jetson.utils.cudaFromNumpy(frame)
    detections = net.Detect(img_cuda, frame.shape[1], frame.shape[0])

    # Draw detection results
    for detection in detections:
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        confidence = detection.Confidence
        class_desc = net.GetClassDesc(detection.ClassID)

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = "{} ({:.2f})".format(class_desc, confidence)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow("Jetson Nano Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
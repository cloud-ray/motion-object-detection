# main.py
import cv2
import time
from utils.detection_manager import DetectionManager
from utils.frame_processing import resize_frame, display_frames
from utils.system_metrics import print_system_metrics
from utils.fps import FPSCounter

# Video stream URL
url = "http://192.168.4.24:8080/video"

# Open video capture from the stream
cap = cv2.VideoCapture(url)

# Initialize detection manager with a motion threshold
detection_manager = DetectionManager(motion_threshold=60)  # Set desired threshold

# Initialize FPS counter
fps_counter = FPSCounter(frame_rate=45)  # Set desired frame rate

# Define a variable for FPS calculation interval
fps_calc_interval = 30  # Calculate and display FPS every 30 frames

# Metrics tracking
processing_times = []

while True:
    # Measure start time for latency calculation
    start_time = time.time()

    # Check if it's time to process a new frame based on FPSCounter
    if not fps_counter.should_process_frame():
        continue

    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Print original frame shape
    original_shape = frame.shape
    # print(f"Original frame shape: {original_shape}")

    # Resize the frame
    frame = resize_frame(frame)

    # Print resized frame shape
    resized_shape = frame.shape
    # print(f"Resized frame shape: {resized_shape}")

    # Process the frame for motion and object detection
    fgMask, motion_detected = detection_manager.process_frame(frame)

    # Print detection info
    detection_manager.print_detection_info()

    # Display the resulting frame (optional)
    display_frames(frame, fgMask)

    # Measure processing time for latency calculation
    processing_time = time.time() - start_time
    processing_times.append(processing_time)

    # Update FPS counter
    fps_counter.update()
    if fps_counter.counter >= fps_calc_interval:  # Calculate FPS every 30 frames
        # Calculate and display FPS
        fps = fps_counter.reset()
        print(f"FPS: {fps:.2f}")

        # Calculate average processing latency
        avg_processing_time = sum(processing_times) / len(processing_times)
        print(f"Average processing latency: {avg_processing_time:.4f} seconds")

        # Reset processing times
        processing_times = []

        # Monitor system metrics
        print_system_metrics()

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

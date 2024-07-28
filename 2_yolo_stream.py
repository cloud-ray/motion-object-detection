# YOLO STREAM WITH PLAIN PREDICT
import cv2
import time
import psutil

import supervision as sv
from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("models/yolov8s-worldv2.pt")
classes = ["lamp", "table", "sofa chair"]
model.set_classes(classes)

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=0.75, text_color=sv.Color.BLACK)

def main():
    stream_url = "http://192.168.4.24:8080/video"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Unable to open video stream at {stream_url}")
        return

    # Attempt to get original stream FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Original stream FPS not available")
    else:
        print(f"Original Stream FPS: {original_fps}")

    skip_frames = 5  # Skip every 'n' frames

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Measure start time for latency calculation
            frame_start_time = time.time()

            # Skip 'skip_frames' number of frames
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret:
                    print("Error: Unable to read frame")
                    break

            # Process the next frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break

            # Print original frame size
            original_height, original_width = frame.shape[:2]
            # print(f"Original Frame Size: {original_width}x{original_height}")

            # Scale down the frame by 50%
            scaled_frame = cv2.resize(frame, (original_width // 2, original_height // 2))
            scaled_height, scaled_width = scaled_frame.shape[:2]
            # print(f"Scaled Frame Size: {scaled_width}x{scaled_height}")

            # Calculate local stream FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            local_fps = frame_count / elapsed_time

            # Calculate frame latency
            frame_latency = (time.time() - frame_start_time)

            results = model.predict(scaled_frame)
            # print("### RESULTS")
            # print(results)

            # Iterate through each result if results is a list
            for result in results:
                # print(result)

                # ########## NO SUPERVISION ##########

                # # Access the boxes attribute
                # boxes = result.boxes

                # # Get the class name mapping
                # class_names = result.names

                # # Iterate through the boxes
                # for i, box in enumerate(boxes):
                #     # print(f"Box {i+1}:")
                #     # Get the class name from the ID
                #     class_name = class_names[box.cls.item()]
                #     # print(f"Class: {class_name}")
                #     # print(f"Class: {box.cls.item()}")
                #     # print(f"Confidence: {box.conf.item():.3f}")
                #     # print(f"ID: {box.id}")
                #     # print(f"Is Track: {box.is_track}")
                #     # print(f"Bounding Box (xyxy): {box.xyxy.tolist()}")
                    
                #     # Create a label
                #     label = f"{class_name} {box.conf.item():.3f}"
                    
                #     # Draw the bounding box
                #     cv2.rectangle(scaled_frame, (int(box.xyxy[0, 0]), int(box.xyxy[0, 1])), (int(box.xyxy[0, 2]), int(box.xyxy[0, 3])), (0, 255, 0), 2)
                    
                #     # Draw the label
                #     cv2.putText(scaled_frame, label, (int(box.xyxy[0, 0]), int(box.xyxy[0, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #     ########## NO SUPERVISION ##########




                ########## SUPERVISION ##########
                detections = sv.Detections.from_ultralytics(result)
                # print("##### Detections")
                # print(detections)
                # print()

                # Create labels with class names and confidence scores
                class_names = detections.data['class_name'].tolist()
                confidences = detections.confidence.tolist()

                labels = [
                    f"{class_name} {confidence:.3f}"
                    for class_name, confidence in zip(class_names, confidences)
                ]

                annotated_frame = scaled_frame.copy()
                
                # Annotate bounding boxes
                annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
                
                # Annotate labels with confidence scores
                annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
                ########## SUPERVISION ##########







                # Display the frame
                cv2.imshow('Stream', annotated_frame)

            # Print performance metrics every 'original_fps' frames
            if frame_count % int(original_fps) == 0:
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_usage = psutil.virtual_memory().percent
                print(f"Local Stream FPS: {local_fps:.2f}, Frame Latency: {frame_latency:.4f}ms, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stream stopped by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


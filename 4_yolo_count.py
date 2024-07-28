# YOLO STREAM WITH TRACK
import cv2
import time
import psutil
import os
import supervision as sv
from ultralytics import YOLOWorld, solutions

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("models/yolov8s-worldv2.pt")
classes = ["lamp", "table", "sofa chair", "person"]
model.set_classes(classes)

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=0.75, text_color=sv.Color.BLACK)

def main():
    stream_url = "http://192.168.4.24:8080/video"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Unable to open video stream at {stream_url}")
        return



    # Get the frame shape
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame shape: {original_width}x{original_height}")

    scaled_region_width = original_width // 2
    scaled_region_height = original_height // 2

    print(f"Scaled Region Sizes: {scaled_region_width}x{scaled_region_height}")

    padding_x = int(scaled_region_width * 0.05)
    padding_y = int(scaled_region_height * 0.05)



    # Attempt to get original stream FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Original stream FPS not available")
    else:
        print(f"Original Stream FPS: {original_fps}")

    skip_frames = 2  # Skip every 'n' frames
    frame_count = 0
    start_time = time.time()


    # Create the region points
    region_points = [
        (padding_x, padding_y),  # top-left
        (scaled_region_width - padding_x, padding_y),  # top-right
        (scaled_region_width - padding_x, scaled_region_height - padding_y),  # bottom-right
        (padding_x, scaled_region_height - padding_y)  # bottom-left
    ]

    # Init Object Counter
    counter = solutions.ObjectCounter(
        view_img=True,
        reg_pts=region_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
        view_in_counts=True,
        view_out_counts=True
    )

    tracker_config_path = os.path.join('utils', 'bytetrack.yaml')


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
            # original_height, original_width = frame.shape[:2]
            # print(f"Original Frame Size: {original_width}x{original_height}")

            # Scale down the frame by 50%
            scaled_frame = cv2.resize(frame, (original_width // 2, original_height // 2))
            # scaled_height, scaled_width = scaled_frame.shape[:2]
            # print(f"Scaled Frame Size: {scaled_width}x{scaled_height}")

            # Calculate local stream FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            local_fps = frame_count / elapsed_time

            # Calculate frame latency
            frame_latency = (time.time() - frame_start_time)





            # Track objects in the frame
            results = model.track(
                source=scaled_frame, # source directory for images or videos
                persist=True, # persisting tracks between frames
                tracker=tracker_config_path, # Tracking method 'bytetrack' or 'botsort'
                conf=0.5, # Confidence Threshold
                iou=0.5, # IOU Threshold
                classes=None, # filter results by class, i.e. classes=0, or classes=[0,2,3]
                verbose=True # Display the object tracking results
            )

            # Use the counter to start counting objects in the frame
            annotated_frame = counter.start_counting(scaled_frame, results)

            # Iterate through each result if results is a list
            for result in results:
                # print(result)

                ########## SUPERVISION ##########
                detections = sv.Detections.from_ultralytics(result)

                # Create labels with class names and confidence scores
                class_names = detections.data['class_name'].tolist()
                confidences = detections.confidence.tolist()
                track_ids = detections.tracker_id.tolist()

                labels = [
                    f"{class_name} (ID: {track_id}) {confidence:.3f}"
                    for class_name, confidence, track_id in zip(class_names, confidences, track_ids)
                ]
                for class_name, confidence, track_id in zip(class_names, confidences, track_ids):
                    print(f"Class Name: {class_name}, Confidence: {confidence:.3f}, Track ID: {track_id}")

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
                print()
                print(f"Local Stream FPS: {local_fps:.2f}, Frame Latency: {frame_latency:.4f}ms, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
                print()

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




                # ########## NO SUPERVISION ##########
                # # Access the boxes attribute
                # boxes = result.boxes

                # # Get the class name mapping
                # class_names = result.names

                # # Iterate through the boxes
                # for i, box in enumerate(boxes):
                #     # print(f"Box {i+1}:")
                #     # print(box)
                #     # Get the class name from the ID
                #     class_name = class_names[box.cls.item()]
                #     # print(f"Class: {class_name}")
                #     # print(f"Class: {box.cls.item()}")
                #     # print(f"Confidence: {box.conf.item():.3f}")
                #     # print(f"Class ID: {box.cls.item()}")
                #     # print(f"Is Track: {box.is_track}")
                #     # print(f"Bounding Box (xyxy): {box.xyxy.tolist()}")
                #     print(f"Box {i+1}: Class={class_names[box.cls.item()]}, Class ID={box.cls.item()}, Confidence={box.conf.item():.3f}, Is Track={box.is_track}, Tracking ID={box.id.item()},")
                    
                #     # Create a label
                #     label = f"{class_name} {box.conf.item():.3f}"
                    
                #     # Draw the bounding box
                #     cv2.rectangle(scaled_frame, (int(box.xyxy[0, 0]), int(box.xyxy[0, 1])), (int(box.xyxy[0, 2]), int(box.xyxy[0, 3])), (0, 255, 0), 2)
                    
                #     # Draw the label
                #     cv2.putText(scaled_frame, label, (int(box.xyxy[0, 0]), int(box.xyxy[0, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #     ########## NO SUPERVISION ##########
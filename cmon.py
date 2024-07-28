# from ultralytics import YOLO
import cv2
import supervision as sv
import time
from inference.models.yolo_world.yolo_world import YOLOWorld

from utils.video_utils import open_video_stream, calculate_fps, display_fps_on_frame

model = YOLOWorld(model_id="yolo_world/s")

classes = ["stuffed animal", "sofa chair", "teapot", "lamp"]
model.set_classes(classes)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)


# Define target dimensions for resizing
TARGET_WIDTH = 960
TARGET_HEIGHT = 540


def process_video_stream(url, skip_frames=5):
    """Main function to process the video stream."""
    cap = open_video_stream(url)

    start_time = time.time()
    frame_count = 0
    processed_frame_count = 0
    total_inference_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break


        # Resize the frame
        resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))


        # Calculate FPS
        frame_count += 1
        fps_avg = calculate_fps(start_time, frame_count)

        # Print the FPS value to the console
        print(f"Overall FPS: {fps_avg:.2f}")

        # Display FPS on the frame
        display_fps_on_frame(frame, fps_avg)

        # Print frame shape and inference results
        print("Original Frame shape:", frame.shape)
        print("Resized Frame shape:", resized_frame.shape)

        skip_frames = 30
        # Skip frames to reduce processing load
        if frame_count % skip_frames != 0:
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Perform inference
        print("Frame shape:", frame.shape)
        inference_start = time.time()

        results = model.infer(resized_frame)


        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        processed_frame_count += 1

        # Calculate and print inference FPS
        inference_fps = processed_frame_count / total_inference_time
        print(f"Inference FPS: {inference_fps:.2f}")

        # Print inference results
        print("Inference results:")
        for result in results.predictions:
            print(f"Class: {result.class_name}, "
                  f"Confidence: {result.confidence:.3f}, "
                  f"Bounding Box: (x: {result.x:.1f}, y: {result.y:.1f}, "
                  f"w: {result.width:.1f}, h: {result.height:.1f})")

        # Process detections
        detections = sv.Detections.from_inference(results)


        # Map detections back to original frame size
        scale_x = frame.shape[1] / TARGET_WIDTH
        scale_y = frame.shape[0] / TARGET_HEIGHT

        original_detections = sv.Detections(
            xyxy=detections.xyxy * [scale_x, scale_y, scale_x, scale_y],
            confidence=detections.confidence,
            class_id=detections.class_id,
            data=detections.data
        )

        print("Detections:")
        for i in range(len(original_detections.confidence)):
            print(f"Class: {original_detections.data['class_name'][i]}, "
                  f"Confidence: {original_detections.confidence[i]:.3f}, "
                  f"Coordinates: {original_detections.xyxy[i]}")

        # Annotate frame
        annotated_frame = frame.copy()

        annotated_frame = box_annotator.annotate(
            annotated_frame,
            original_detections
        )

        labels = [
            f"{original_detections.data['class_name'][i]} {original_detections.confidence[i]:.3f}"
            for i in range(len(original_detections.confidence))
        ]

        annotated_frame = label_annotator.annotate(
            annotated_frame,
            original_detections,
            labels
        )

        # Display the resulting frame
        cv2.imshow('Video Stream', annotated_frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_url = "http://192.168.4.24:8080/video"
    process_video_stream(video_url)





# model = YOLO("./models/yolov8m-world.pt")

# model.set_classes(["stuffed animal", "sandal", "teapot"])  

# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# url = "http://192.168.4.24:8080/video"
# cap = cv2.VideoCapture(url)
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Reduce the resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# target_fps = 1 # 1 frame per second
# # target_fps = 0.5 # 1 frame per 2 seconds
# # target_fps = 0.33 # 1 frame per 3 seconds


# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     start_time = time.time()

#     results = model.predict(frame)

#     detections = sv.Detections.from_ultralytics(results[0])

#     annotated_frame = box_annotator.annotate(
#         scene=frame.copy(),
#         detections=detections
#     )

#     labels = [
#         f"{detections.data['class_name'][i]} {detections.confidence[i]:.3f}"
#         for i in range(len(detections.confidence))
#     ]

#     annotated_frame = label_annotator.annotate(
#         scene=annotated_frame,
#         detections=detections,
#         labels=labels
#     )

#     end_time = time.time()

#     fps = 1 / (end_time - start_time)
#     print(f"FPS: {fps:.2f}")

#     cv2.imshow("Frame", annotated_frame)

#     # Introduce a delay to control the frame rate
#     delay_ms = int(1000 / target_fps)  # Convert target FPS to delay in ms
#     cv2.waitKey(delay_ms)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
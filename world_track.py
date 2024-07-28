import cv2
import supervision as sv
from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("models/yolov8s-worldv2.pt")
classes = ["person", "TV", "chair"]
model.set_classes(classes)

stream = "http://192.168.4.24:8080/video"
cap = cv2.VideoCapture(stream)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    results = model.predict(frame)

    # Iterate through each result if results is a list
    for result in results:

        # PRINT SPEED
        print("Speed Statistics:")
        print("Preprocess: {:.3f}".format(result.speed['preprocess']))
        print("Inference: {:.3f}".format(result.speed['inference']))
        print("Postprocess: {:.3f}".format(result.speed['postprocess']))
        print()
        
        # Access attributes for each result
        boxes = result.boxes  # Bounding boxes in [x1, y1, x2, y2] format
        names = result.names  # Class names dictionary
        probs = result.probs  # Probabilities for classification tasks
        speed = result.speed  # Inference speed

        if boxes is not None:
            for i, box in enumerate(boxes):
                # Extract information for each bounding box
                conf = probs[i].max().item()  # Maximum probability as confidence
                cls = probs[i].argmax().item()  # Class index with maximum probability
                is_track = False  # Assume no tracking information if not available

                # Get class name from the class index
                class_name = names.get(int(cls), "unknown")

                # Print information for the current result
                print(f"Object {i + 1}:")
                print(f"  Class ID: {cls}")
                print(f"  Confidence: {conf}")
                print(f"  Tracking: {is_track}")
                print(f"  Class Name: {class_name}")
                print(f"  Bounding Box: {box.tolist()}")
                print()

    print("Speed:")
    for key, value in speed.items():
        print(f"  {key}: {value}")

    cv2.imshow('Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







# BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
# LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=Color.BLACK)

        # # Create labels with class names and confidence scores
        # labels = [
        #     f"{detection.class_name} {detection.confidence:.3f}"
        #     for detection in detections
        # ]

        # annotated_frame = frame.copy()
        
        # # Annotate bounding boxes
        # annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
        
        # # Annotate labels with confidence scores
        # annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)



# tracker_config_path = os.path.join('utils', 'bytetrack.yaml')

# # Track objects in the frame
# results = model.track(
#     source=frame, # source directory for images or videos
#     persist=True, # persisting tracks between frames
#     tracker=tracker_config_path, # Tracking method 'bytetrack' or 'botsort'
#     conf=0.5, # Confidence Threshold
#     iou=0.5, # IOU Threshold
#     classes=14, # filter results by class, i.e. classes=0, or classes=[0,2,3]
#     verbose=True # Display the object tracking results
# )


#     detections = sv.Detections.from_(results).with_nms(threshold=nms_threshold)



# source = "http://192.168.4.24:8080/video"
# cap = cv2.VideoCapture(source)

# if not cap.isOpened():
#     print("Unable to open video source")
#     exit()

# # Get the video resolution from the live stream
# ret, frame = cap.read()
# height, width, _ = frame.shape
# video_info = sv.VideoInfo(width, height, fps=30)  # assume 30 fps for live stream

# frame_generator = lambda: iter(lambda: cap.read()[1], None)

# BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
# LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

# # Confidence threshold: minimum confidence score required for a detection to be considered valid
# # Lower values (e.g. 0.01) will detect more objects, but may include false positives
# # Higher values (e.g. 0.5) will detect fewer objects, but with higher accuracy
# confidence_threshold = 0.01

# # Non-maximum suppression (NMS) threshold: controls how similar two detections need to be to be merged
# # Lower values (e.g. 0.01) will merge very similar detections, reducing the number of objects detected
# # Higher values (e.g. 0.5) will merge detections that are quite dissimilar, resulting in more objects detected
# nms_threshold = 0.01

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model.infer(frame, confidence=confidence_threshold)
#     detections = sv.Detections.from_inference(results).with_nms(threshold=nms_threshold)
#     detections = detections[(detections.area / (width * height)) < 0.10]

#     # Create labels with class names and confidence scores
#     labels = [
#         f"{detections.data['class_name'][i]} {detections.confidence[i]:.3f}"
#         for i in range(len(detections.confidence))
#     ]

#     annotated_frame = frame.copy()
    
#     # Annotate bounding boxes
#     annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
    
#     # Annotate labels with confidence scores
#     annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

#     cv2.imshow('frame', annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
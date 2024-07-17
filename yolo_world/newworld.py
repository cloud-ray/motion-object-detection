import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

model = YOLOWorld(model_id="yolo_world/s")

classes = ["person", "dog", "chair", "woman"]
model.set_classes(classes)

source = "http://192.168.4.24:8080/video"
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Unable to open video source")
    exit()

# Get the video resolution from the live stream
ret, frame = cap.read()
height, width, _ = frame.shape
video_info = sv.VideoInfo(width, height, fps=30)  # assume 30 fps for live stream

frame_generator = lambda: iter(lambda: cap.read()[1], None)

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

# Confidence threshold: minimum confidence score required for a detection to be considered valid
# Lower values (e.g. 0.01) will detect more objects, but may include false positives
# Higher values (e.g. 0.5) will detect fewer objects, but with higher accuracy
confidence_threshold = 0.01

# Non-maximum suppression (NMS) threshold: controls how similar two detections need to be to be merged
# Lower values (e.g. 0.01) will merge very similar detections, reducing the number of objects detected
# Higher values (e.g. 0.5) will merge detections that are quite dissimilar, resulting in more objects detected
nms_threshold = 0.01

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.infer(frame, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(threshold=nms_threshold)
    detections = detections[(detections.area / (width * height)) < 0.10]

    # Create labels with class names and confidence scores
    labels = [
        f"{detections.data['class_name'][i]} {detections.confidence[i]:.3f}"
        for i in range(len(detections.confidence))
    ]

    annotated_frame = frame.copy()
    
    # Annotate bounding boxes
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
    
    # Annotate labels with confidence scores
    annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

    cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
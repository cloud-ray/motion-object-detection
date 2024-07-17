import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

model = YOLOWorld(model_id="yolo_world/s")

classes = ["lamp", "chair", "table", "dog"]
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.infer(frame, confidence=0.002)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
    detections = detections[(detections.area / (width * height)) < 0.10]

    annotated_frame = frame.copy()
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
    
    cv2.imshow('frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
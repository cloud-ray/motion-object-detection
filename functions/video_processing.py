import cv2
from ultralytics import YOLO

model = YOLO("../models/yolov10n.pt")

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def process_contours(frame, backSub):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    fgMask = backSub.apply(gray)
    thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_motion(contours, min_area=1000):
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            return True
    return False

def track_objects(frame):
    results = model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
    return results

def encode_frame(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

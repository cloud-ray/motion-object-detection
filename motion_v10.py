from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response

# Load a pretrained YOLOv10n model
model = YOLO("./models/yolov10n.pt")

# Background subtractor for movement detection
back_sub = cv2.createBackgroundSubtractorMOG2()

# Define threshold for movement detection (adjust as needed)
threshold_limit = 1000  # Experiment to find the optimal value

# Number of frames to skip for background learning
skip_frames = 10  # Adjust as needed

# Set the motion detection threshold
motion_threshold = 500

# Flask app setup
app = Flask(__name__)

# RTSP stream URL (replace with your IP camera's URL)
source = "http://192.168.4.24:8080/video"

# Generate video stream from IP camera and perform YOLOv10 inference
def gen_video_stream():
    cap = cv2.VideoCapture(source)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Perform inference on the frame
        results = model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")

        # Filter out static objects
        moving_objects = []
        for result in results:
            annotated_frame = result.plot()  # Annotate the frame with detection results
            for *box, conf, cls in result.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                # Check if the detected object has motion
                if fg_mask[y1:y2, x1:x2].mean() > 25:  # Adjust the threshold value as needed
                    moving_objects.append((x1, y1, x2, y2, conf, cls))

        # Draw rectangles around moving objects
        for x1, y1, x2, y2, _, _ in moving_objects:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    return Response(gen_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=8079)

from ultralytics import YOLO
import cv2
from flask import Flask, render_template, Response

# Load a pretrained YOLOv10n model
model = YOLO("./models/yolov10n.pt")

# Flask app setup
app = Flask(__name__)

# RTSP stream URL (replace with your IP camera's URL)
source = "http://192.168.4.24:8080/video"

# Generate video stream from IP camera and perform YOLOv10 inference
def gen_video_stream():
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame
        # results = model(frame, stream=True)
        results = model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")


        # Process and display results
        for result in results:
            annotated_frame = result.plot()  # Annotate the frame with detection results

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

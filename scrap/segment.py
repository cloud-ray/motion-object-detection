from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model with segmentation capabilities
model = YOLO("./models/yolov8n-seg.pt")

# IP camera source
source = "http://192.168.4.24:8080/video"  # Replace with your IP camera address
cap = cv2.VideoCapture(source)

# Function to generate frames for streaming
def generate_frames():
    while True:
        # Read a frame from the video capture
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Create an annotator object to draw on the frame
        annotator = Annotator(im0, line_width=2)

        # Perform object tracking on the current frame
        results = model.track(im0, persist=True)

        # Check if tracking IDs and masks are present in the results
        if results[0].boxes.id is not None and results[0].masks is not None:
            # Extract masks, tracking IDs, and class IDs
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get class IDs

            for i, (mask, track_id, class_id) in enumerate(zip(masks, track_ids, class_ids)):
                annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True)) 
                x, y = int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][3])
                label = f"{model.names[class_id]} {track_id}"  # Create label string
                annotator.text((x, y), label, txt_color=(255, 255, 255))

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define app routes
@app.route('/')
def index():
    return render_template('index.html')  # Create an index.html file

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
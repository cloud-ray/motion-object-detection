from flask import Flask, Response, render_template
import cv2
from collections import defaultdict

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

track_history = defaultdict(lambda: [])

model = YOLO("./models/yolov8n-seg.pt")  # segmentation model

# Replace with your IP camera URL
source = "http://192.168.4.24:8080/video"
cap = cv2.VideoCapture(source)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        annotator = Annotator(im0, line_width=3)

        results = model.track(im0, persist=True)

        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()  # Get the class indices

            for mask, track_id, class_index in zip(masks, track_ids, class_indices):
                class_name = model.model.names[class_index]  # Get the class name from the index
                color = colors(int(track_id), True)
                txt_color = annotator.get_txt_color(color)
                annotator.seg_bbox(mask=mask, mask_color=color, label=f"ID{track_id}:{class_name}", txt_color=txt_color)

        ret, jpeg = cv2.imencode('.jpg', im0)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)








# import cv2
# from collections import defaultdict

# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

# track_history = defaultdict(lambda: [])

# model = YOLO("./models/yolov8n-seg.pt")  # segmentation model

# # Replace with your IP camera URL
# source = "http://192.168.4.24:8080/video"
# cap = cv2.VideoCapture(source)

# while True:
#     ret, im0 = cap.read()
#     if not ret:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     annotator = Annotator(im0, line_width=4)

#     results = model.track(im0, persist=True)

#     if results[0].boxes.id is not None and results[0].masks is not None:
#         masks = results[0].masks.xy
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#         class_indices = results[0].boxes.cls.int().cpu().tolist()  # Get the class indices

#         for mask, track_id, class_index in zip(masks, track_ids, class_indices):
#             class_name = model.model.names[class_index]  # Get the class name from the index
#             color = colors(int(track_id), True)
#             txt_color = annotator.get_txt_color(color)
#             annotator.seg_bbox(mask=mask, mask_color=color, label=f"ID{track_id}:{class_name}", txt_color=txt_color)

#     cv2.imshow("instance-segmentation-object-tracking", im0)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
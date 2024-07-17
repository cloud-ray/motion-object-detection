from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import time

model = YOLO("./models/yolov10n.pt")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


def gen_video_stream():
    url = "http://192.168.4.24:8080/video"
    cap = cv2.VideoCapture(url)
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_rate = 2  # adjust this value to change the frame subsampling rate
    frame_counter = 0
    motion_start_time = None
    motion_end_time = None
    no_motion_start_time = None
    no_motion_end_time = None
    detection_summary = {}  # dictionary to store unique IDs for each class

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgMask = backSub.apply(gray)
        thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # adjust this value to filter out small contours
                motion_detected = True
                break

        if motion_detected:
            if motion_start_time is None:
                motion_start_time = time.time()
            frame_counter += 1
            if frame_counter % frame_rate == 0:  # skip frames based on the specified rate
                results = model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
                for result in results:
                    annotated_frame = result.plot()  # Annotate the frame with detection results
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # Update detection summary
                    for i, box in enumerate(result.boxes):
                        class_name = result.names[int(box.cls)]
                        id = i
                        if class_name not in detection_summary:
                            detection_summary[class_name] = set()
                        detection_summary[class_name].add(id)

            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            if no_motion_start_time is None:
                no_motion_start_time = time.time()
            if motion_start_time is not None:
                motion_end_time = time.time()
                print(f"Motion detected from {motion_start_time} to {motion_end_time} ({motion_end_time - motion_start_time:.2f} seconds)")
                motion_start_time = None
                # Print detection summary
                for class_name, ids in detection_summary.items():
                    print(f"Class: {class_name}, Unique IDs: {len(ids)}")
                detection_summary = {}  # reset detection summary
            print("No detection. Skipping YOLO...")
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if no_motion_start_time is not None and motion_detected:
            no_motion_end_time = time.time()
            print(f"No motion detected from {no_motion_start_time} to {no_motion_end_time} ({no_motion_end_time - no_motion_start_time:.2f} seconds)")
            no_motion_start_time = None


@app.route("/video")
def video_feed():
    return Response(gen_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=8079)




# # Lower Framerate Working Detection
# def gen_video_stream():
#     url = "http://192.168.4.24:8080/video"
#     cap = cv2.VideoCapture(url)
#     backSub = cv2.createBackgroundSubtractorMOG2()
#     frame_rate = 2  # adjust this value to change the frame subsampling rate
#     frame_counter = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)
#         fgMask = backSub.apply(gray)
#         thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         motion_detected = False
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > 1000:  # adjust this value to filter out small contours
#                 motion_detected = True
#                 break
#         if motion_detected:
#             frame_counter += 1
#             if frame_counter % frame_rate == 0:  # skip frames based on the specified rate
#                 results = model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
#                 for result in results:
#                     annotated_frame = result.plot()  # Annotate the frame with detection results
#                     _, buffer = cv2.imencode('.jpg', annotated_frame)
#                     frame = buffer.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             else:
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             print("No detection. Skipping YOLO...")
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# # Basic Working Detection
# def gen_video_stream():
#     url = "http://192.168.4.24:8080/video"
#     cap = cv2.VideoCapture(url)
#     prev_frame = None
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)
#         if prev_frame is None:
#             prev_frame = gray
#             continue
#         frame_delta = cv2.absdiff(prev_frame, gray)
#         thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
#         if cv2.countNonZero(thresh) > 0:
#             print("Motion detected!")
#         else:
#             print("No detection.")
#         prev_frame = gray
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# # More Advanced Working Detection
# def gen_video_stream():
#     url = "http://192.168.4.24:8080/video"
#     cap = cv2.VideoCapture(url)
#     backSub = cv2.createBackgroundSubtractorMOG2()
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)
#         fgMask = backSub.apply(gray)
#         thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > 1000:  # adjust this value to filter out small contours
#                 print("Motion detected!")
#                 break
#         else:
#             print("No detection.")
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



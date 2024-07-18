from flask import Flask, render_template, Response
import cv2
import time
import threading
import queue
import cProfile
from functions.video_processing import process_contours, detect_motion, track_objects, encode_frame

app = Flask(__name__)

frame_queue = queue.Queue(maxsize=10)  # Adjust the queue size as needed
 
def frame_capture(url, frame_queue):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  # Small sleep to avoid busy-waiting

@app.route("/")
def index():
    return render_template("index.html")

def gen_video_stream():
    url = "http://192.168.4.24:8080/video"
    threading.Thread(target=frame_capture, args=(url, frame_queue)).start()
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_rate = 2  # adjust this value to change the frame subsampling rate
    frame_counter = 0
    motion_start_time = None
    detection_summary = {}  # dictionary to store unique IDs for each class

    while True:
        frame = frame_queue.get()
        contours = process_contours(frame, backSub)
        motion_detected = detect_motion(contours)

        if motion_detected:
            if motion_start_time is None:
                motion_start_time = time.time()
            frame_counter += 1
            if frame_counter % frame_rate == 0:  # skip frames based on the specified rate
                results = track_objects(frame)
                for result in results:
                    annotated_frame = result.plot()  # Annotate the frame with detection results
                    frame = encode_frame(annotated_frame)
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
                frame = encode_frame(frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            frame = encode_frame(frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if motion_start_time is not None and not motion_detected:
            motion_end_time = time.time()
            print(f"Motion detected from {motion_start_time} to {motion_end_time} ({motion_end_time - motion_start_time:.2f} seconds)")
            motion_start_time = None
            # Print detection summary
            for class_name, ids in detection_summary.items():
                print(f"Class: {class_name}, Unique IDs: {len(ids)}")
            detection_summary = {}  # reset detection summary

@app.route("/video")
def video_feed():
    return Response(gen_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    app.run(debug=True, port=8079)
    profiler.disable()
    profiler.dump_stats("./logs/profile.log")

# app.py
import logging
from flask import Flask, render_template, Response
from functions.motion_detector import MotionDetector
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Configure logging to save logs to a file and append to it
logging.basicConfig(filename='./logs/app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    filemode='a')  # Ensure appending mode
logger = logging.getLogger(__name__)

# Load YOLO model
yolo_model = YOLO("./models/yolov10n.pt")
### LOG ###
logger.info("YOLO model loaded successfully.")

# Create an instance of the MotionDetector class
detector = MotionDetector("http://192.168.4.24:8080/video", yolo_model)
### LOG ###
logger.info("MotionDetector instance created.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        ret, frame = detector.cap.read()
        if not ret:
            ### LOG ###
            logger.error("Failed to read frame from camera.")
            break

        motion_detected, frame = detector.detect_motion(frame)
        detector.log_motion(motion_detected)

        if motion_detected:
            ### LOG ###
            logger.info("Motion detected.")
            print("Motion detected!")
            frame = detector.yolo_detector.detect_and_track(frame)
        else:
            ### LOG ###
            logger.debug("No motion detected.")
            print("No motion detected.")

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            ### LOG ###
            logger.error("Failed to encode frame to JPEG.")
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

if __name__ == '__main__':
    try:
        app.run(debug=True, port=8079)
        ### LOG ###
        logger.info("Flask app started on port 8079.")
    except Exception as e:
        ### LOG ###
        logger.critical("Critical error: %s", e)
    finally:
        detector.cap.release()
        cv2.destroyAllWindows()
        ### LOG ###
        logger.info("Released camera and destroyed all windows.")









# # FUNCTIONING WITHOUT YOLO
# from flask import Flask, render_template, Response
# from functions.motion_detector import MotionDetector
# import cv2

# app = Flask(__name__)

# # Create an instance of the MotionDetector class
# detector = MotionDetector("http://192.168.4.24:8080/video")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames():
#     while True:
#         ret, frame = detector.cap.read()
#         if not ret:
#             break

#         motion_detected, frame = detector.detect_motion(frame)

#         if motion_detected:
#             print("Motion detected!")
#         else:
#             print("No motion detected.")

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# if __name__ == '__main__':
#     try:
#         app.run(debug=True)
#     finally:
#         detector.cap.release()
#         cv2.destroyAllWindows()








# FUNCTIONING OPTIMIZED 
#         app = Flask(__name__)

# # Create an instance of the MotionDetector class
# detector = MotionDetector("http://192.168.4.24:8080/video")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames():
#     while True:
#         ret, frame = detector.cap.read()
#         if not ret:
#             break

#         motion_detected, frame = detector.detect_motion(frame)

#         if motion_detected:
#             print("Motion detected!")
#         else:
#             print("No motion detected.")

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# if __name__ == '__main__':
#     try:
#         app.run(debug=True)
#     finally:
#         detector.cap.release()
#         cv2.destroyAllWindows()






# FUNCTION DETECTOR W/O MODEL
    # def gen_frames():
    # while True:
    #     ret, frame = detector.cap.read()
    #     if not ret:
    #         break

    #     fg_mask = detector.apply_background_subtraction(frame)
    #     fg_mask = detector.apply_morphological_opening(fg_mask)
    #     contours, _ = detector.find_contours(fg_mask)
    #     motion_detected = detector.process_contours(contours, frame)

    #     if motion_detected:
    #         print("Motion detected!")
    #     else:
    #         print("No motion detected.")

    #     ret, jpeg = cv2.imencode('.jpg', frame)
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')









# FUNCTIONING MOTION DETECTOR
# import cv2
# import numpy as np

# url = "http://192.168.4.24:8080/video"
# cap = cv2.VideoCapture(url)

# fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)

# minArea = 750

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Background subtraction
#     fgMask = fgbg.apply(frame)

#     # Create a 5x5 kernel for morphological operations
#     kernel = np.ones((5, 5), np.uint8)

#     # Apply morphological opening to reduce noise
#     fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

#     # Find outer contours in the foreground mask
#     contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     motion_detected = False

#     # Filter and process substantial movement
#     for contour in contours:
#         # ignore small contours
#         if cv2.contourArea(contour) < minArea:
#             continue

#         motion_detected = True

#         # draw bounding box around the detected object
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     if motion_detected:
#         print("Motion detected!")
#     else:
#         print("No motion detected.")

#     cv2.imshow('Frame', frame)
#     cv2.imshow('Foreground Mask', fgMask)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

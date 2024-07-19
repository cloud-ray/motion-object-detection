# functions/motion_detector.py
import logging
import cv2
import numpy as np
from functions.yolo_detector import YoloDetector
import time


class MotionDetector:
    # def __init__(self, url, yolo_model, min_area=750):
    def __init__(self, stream, yolo_model, min_area=750):

        # # IP Camera
        # self.cap = cv2.VideoCapture(url)
        # YouTube Live
        self.cap = stream  # Use the stream directly

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
        self.min_area = min_area
        self.yolo_detector = YoloDetector(yolo_model)
        self.motion_start_time = None
        self.no_motion_start_time = time.time()

        # Configure logging to save logs to a file and append to it
        logging.basicConfig(filename='./logs/motion_detector.log', level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                            filemode='a')  # Ensure appending mode
        self.logger = logging.getLogger(__name__)
        self.logger.info("MotionDetector initialized.")

    def apply_background_subtraction(self, frame):
        return self.fgbg.apply(frame)

    def apply_morphological_opening(self, fg_mask):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    def find_contours(self, fg_mask):
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_motion(self, frame):
        fg_mask = self.apply_background_subtraction(frame)
        fg_mask = self.apply_morphological_opening(fg_mask)
        contours = self.find_contours(fg_mask)
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return motion_detected, frame

    def log_motion(self, motion_detected):
        current_time = time.time()
        if motion_detected:
            if self.no_motion_start_time is not None:
                no_motion_duration = current_time - self.no_motion_start_time
                ### LOG ###
                self.logger.info(f"No motion detected for {no_motion_duration:.2f} seconds.")
                print(f"No motion detected for {no_motion_duration:.2f} seconds.")
                self.no_motion_start_time = None
            if self.motion_start_time is None:
                self.motion_start_time = current_time
        else:
            if self.motion_start_time is not None:
                motion_duration = current_time - self.motion_start_time
                ### LOG ###
                self.logger.info(f"Motion detected for {motion_duration:.2f} seconds.")
                print(f"Motion detected for {motion_duration:.2f} seconds.")
                self.yolo_detector.summarize_detections()  # Summarize detections when motion stops
                self.yolo_detector.detections = []  # Clear detections after summarizing
                self.motion_start_time = None
            if self.no_motion_start_time is None:
                self.no_motion_start_time = current_time

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                ### LOG ###
                self.logger.error("Failed to read frame from camera.")
                break

            motion_detected, frame = self.detect_motion(frame)
            self.log_motion(motion_detected)

            if motion_detected:
                annotated_frame = self.yolo_detector.detect_and_track(frame)
                cv2.imshow('Frame', annotated_frame)
            else:
                ### LOG ###
                self.logger.debug("No motion detected.")
                print("No motion detected.")
                cv2.imshow('Frame', frame)

            cv2.imshow('Foreground Mask', self.apply_background_subtraction(frame))

            if cv2.waitKey(1) == ord('q'):
                ### LOG ###
                self.logger.info("Quitting the application.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        ### LOG ###
        self.logger.info("Released camera and destroyed all windows.")

if __name__ == "__main__":
    from ultralytics import YOLO
    yolo_model = YOLO("./models/yolov10n.pt")
    detector = MotionDetector("http://192.168.4.24:8080/video", yolo_model)
    detector.run()














# # FUNCTIONING MOTION DETECTOR
# class MotionDetector:
#     def __init__(self, url, yolo_model, min_area=750):
#         self.cap = cv2.VideoCapture(url)
#         self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
#         self.min_area = min_area
#         self.yolo_detector = YoloDetector(yolo_model)
#         self.motion_start_time = None
#         self.no_motion_start_time = time.time()

#     def apply_background_subtraction(self, frame):
#         return self.fgbg.apply(frame)

#     def apply_morphological_opening(self, fg_mask):
#         kernel = np.ones((5, 5), np.uint8)
#         return cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

#     def find_contours(self, fg_mask):
#         return cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     def detect_motion(self, frame):
#         fg_mask = self.apply_background_subtraction(frame)
#         fg_mask = self.apply_morphological_opening(fg_mask)
#         contours, _ = self.find_contours(fg_mask)
#         motion_detected = False
#         for contour in contours:
#             if cv2.contourArea(contour) < self.min_area:
#                 continue
#             motion_detected = True
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         return motion_detected, frame

#     def log_motion(self, motion_detected):
#         current_time = time.time()
#         if motion_detected:
#             if self.no_motion_start_time is not None:
#                 no_motion_duration = current_time - self.no_motion_start_time
#                 print(f"No motion detected for {no_motion_duration:.2f} seconds.")
#                 self.no_motion_start_time = None
#             if self.motion_start_time is None:
#                 self.motion_start_time = current_time
#         else:
#             if self.motion_start_time is not None:
#                 motion_duration = current_time - self.motion_start_time
#                 print(f"Motion detected for {motion_duration:.2f} seconds.")
#                 # print(f"Object(s) Detected: {self.yolo_detector.get_summary()}")
#                 self.motion_start_time = None
#             if self.no_motion_start_time is None:
#                 self.no_motion_start_time = current_time

#     def run(self):
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             motion_detected, frame = self.detect_motion(frame)
#             self.log_motion(motion_detected)

#             if motion_detected:
#                 # print("Motion detected!")
#                 annotated_frame = self.yolo_detector.detect_and_track(frame)
#                 cv2.imshow('Frame', annotated_frame)
#             else:
#                 print("No motion detected.")
#                 cv2.imshow('Frame', frame)

#             cv2.imshow('Foreground Mask', self.apply_background_subtraction(frame))

#             if cv2.waitKey(1) == ord('q'):
#                 break

# if __name__ == "__main__":
#     from ultralytics import YOLO
#     yolo_model = YOLO("./models/yolov10n.pt")
#     detector = MotionDetector("http://192.168.4.24:8080/video", yolo_model)
#     detector.run()


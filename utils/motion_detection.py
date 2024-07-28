# utils/motion_detection.py
import cv2
import time

class MotionDetector:
    def __init__(self, frame_rate=5):
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.frame_rate = frame_rate
        self.frame_counter = 0
        self.motion_start_time = None
        self.no_motion_start_time = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgMask = self.backSub.apply(gray)
        thresh = cv2.threshold(fgMask, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = any(cv2.contourArea(contour) > 2000 for contour in contours)

        if motion_detected:
            if self.motion_start_time is None:
                self.motion_start_time = time.time()
            self.no_motion_start_time = None
            self.frame_counter += 1
        else:
            if self.no_motion_start_time is None:
                self.no_motion_start_time = time.time()
            if self.motion_start_time is not None:
                self.motion_start_time = None

        return fgMask, motion_detected

    def print_motion_info(self):
        if self.motion_start_time is not None:
            motion_end_time = time.time()
            print(f"Motion detected from {self.motion_start_time:.2f} to {motion_end_time:.2f} "
                  f"({motion_end_time - self.motion_start_time:.2f} seconds)")
        if self.no_motion_start_time is not None:
            no_motion_end_time = time.time()
            print(f"No motion detected from {self.no_motion_start_time:.2f} to {no_motion_end_time:.2f} "
                  f"({no_motion_end_time - self.no_motion_start_time:.2f} seconds)")

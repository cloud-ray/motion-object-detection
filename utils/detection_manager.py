# utils/detection_manager.py
from utils.motion_detection import MotionDetector
from utils.object_detection import ObjectDetector

class DetectionManager:
    def __init__(self, motion_threshold=60):
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector()  # Initialize your object detector
        self.motion_threshold = motion_threshold
        self.consecutive_motion_frames = 0

    def process_frame(self, frame):
        fgMask, motion_detected = self.motion_detector.process_frame(frame)

        if motion_detected:
            self.consecutive_motion_frames += 1
            print(f"Motion detected in consecutive frames: {self.consecutive_motion_frames}")
            if self.consecutive_motion_frames >= self.motion_threshold:
                self.trigger_object_detection(frame)
                # Reset counter after triggering object detection
                self.consecutive_motion_frames = 0
        else:
            if self.consecutive_motion_frames > 0:
                print(f"Motion was detected for {self.consecutive_motion_frames} frames but did not reach the threshold of {self.motion_threshold} frames.")
            self.consecutive_motion_frames = 0

        return fgMask, motion_detected

    def print_detection_info(self):
        self.motion_detector.print_motion_info()

    def trigger_object_detection(self, frame):
        # Trigger the actual object detection logic
        print("########## Object detection model triggered ##########")
        detected_objects = self.object_detector.detect_objects(frame)
        print(f"Detected objects: {detected_objects}")

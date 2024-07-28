# utils/fps.py
import time

class FPSCounter:
    def __init__(self, frame_rate=5):
        self.frame_rate = frame_rate
        self.counter = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()

    def should_process_frame(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time
        if elapsed_time >= 1.0 / self.frame_rate:
            self.last_frame_time = current_time
            return True
        return False

    def update(self):
        self.counter += 1

    def get_fps(self):
        end_time = time.time()
        if self.counter == 0:
            return 0
        return self.counter / (end_time - self.start_time)

    def reset(self):
        fps = self.get_fps()
        self.counter = 0
        self.start_time = time.time()
        return fps

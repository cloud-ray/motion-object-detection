# utils/frame_processing.py
import cv2

def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, (0, 0), fx=scale, fy=scale)

def display_frames(frame, fgMask):
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)

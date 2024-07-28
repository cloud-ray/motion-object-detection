import cv2
import time

def open_video_stream(url):
    """Open the video stream and return the VideoCapture object."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened(): 
        print("Error: Could not open video stream.")
        exit()
    return cap

def calculate_fps(start_time, frame_count):
    """Calculate the average FPS since the start_time."""
    current_time = time.time()
    fps_avg = frame_count / (current_time - start_time)
    return fps_avg

def display_fps_on_frame(frame, fps_avg):
    """Display the FPS on the given frame."""
    fps_text = f"FPS: {fps_avg:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_width, text_height = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]
    margin = 5

    # Get the frame height and width
    (frame_height, frame_width) = frame.shape[:2]

    # Calculate the top-left corner of the rectangle and text
    x = int(frame_width * 0.05)
    y = int(frame_height * 0.05)

    # Draw a white box behind the text
    cv2.rectangle(frame, (x, y - text_height - margin), (x + text_width + margin, y + margin), (255, 255, 255), -1)

    # Draw the text in black
    cv2.putText(frame, fps_text, (x + margin, y), font, font_scale, (0, 0, 0), font_thickness)

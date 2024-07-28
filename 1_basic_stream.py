import cv2
import time
import psutil

def main():
    stream_url = "http://192.168.4.24:8080/video"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Unable to open video stream at {stream_url}")
        return

    # Attempt to get original stream FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Original stream FPS not available")
    else:
        print(f"Original Stream FPS: {original_fps}")

    skip_frames = 1  # Skip every 'n' frames

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Measure start time for latency calculation
            frame_start_time = time.time()

            # Skip 'skip_frames' number of frames
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret:
                    print("Error: Unable to read frame")
                    break

            # Process the next frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break

            # Calculate local stream FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            local_fps = frame_count / elapsed_time

            # Calculate frame latency
            frame_latency = (time.time() - frame_start_time)

            # Display the frame
            cv2.imshow('Stream', frame)

            # Print performance metrics every 'original_fps' frames
            if frame_count % int(original_fps) == 0:
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_usage = psutil.virtual_memory().percent
                print(f"Local Stream FPS: {local_fps:.2f}, Frame Latency: {frame_latency:.4f}ms, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stream stopped by user")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()










# import cv2
# import time
# import psutil
# from line_profiler import profile

# @profile
# def initialize_video_capture(stream_url):
#     """Initialize video capture from the stream URL."""
#     cap = cv2.VideoCapture(stream_url)
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open video stream at {stream_url}")
#     return cap

# @profile
# def get_original_fps(cap):
#     """Retrieve the original FPS of the video stream."""
#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     if original_fps == 0:
#         print("Original stream FPS not available")
#     else:
#         print(f"Original Stream FPS: {original_fps}")
#     return original_fps

# @profile
# def read_frame(cap, skip_frames):
#     """Read and return a frame from the video stream."""
#     for _ in range(skip_frames):
#         ret, _ = cap.read()
#         if not ret:
#             raise Exception("Error: Unable to read frame")
#     ret, frame = cap.read()
#     if not ret:
#         raise Exception("Error: Unable to read frame")
#     return frame

# @profile
# def calculate_metrics(start_time, frame_start_time, frame_count):
#     """Calculate FPS, frame latency, CPU usage, and memory usage."""
#     elapsed_time = time.time() - start_time
#     local_fps = frame_count / elapsed_time
#     frame_latency = (time.time() - frame_start_time)
#     cpu_usage = psutil.cpu_percent(interval=None)
#     memory_usage = psutil.virtual_memory().percent
#     return local_fps, frame_latency, cpu_usage, memory_usage

# @profile
# def display_frame(frame):
#     """Display the current video frame."""
#     cv2.imshow('Stream', frame)

# @profile
# def print_metrics(local_fps, frame_latency, cpu_usage, memory_usage):
#     """Print the performance metrics."""
#     print(f"Local Stream FPS: {local_fps:.2f}, Frame Latency: {frame_latency:.4f}ms, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

# @profile
# def release_resources(cap):
#     """Release video capture and destroy all windows."""
#     cap.release()
#     cv2.destroyAllWindows()

# @profile
# def main():
#     stream_url = "http://192.168.4.24:8080/video"
#     skip_frames = 1  # Skip every 'n' frames

#     try:
#         cap = initialize_video_capture(stream_url)
#         original_fps = get_original_fps(cap)

#         frame_count = 0
#         start_time = time.time()

#         while True:
#             frame_start_time = time.time()

#             # Read and process a frame
#             frame = read_frame(cap, skip_frames)

#             # Calculate metrics
#             frame_count += 1
#             local_fps, frame_latency, cpu_usage, memory_usage = calculate_metrics(
#                 start_time, frame_start_time, frame_count
#             )

#             # Display the frame
#             display_frame(frame)

#             # Print performance metrics every 'original_fps' frames
#             if original_fps > 0 and frame_count % int(original_fps) == 0:
#                 print_metrics(local_fps, frame_latency, cpu_usage, memory_usage)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     except KeyboardInterrupt:
#         print("Stream stopped by user")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         release_resources(cap)

# if __name__ == "__main__":
#     main()

# Tracks and counts objects in region
# Prints frame an dobject details
import os
import cv2
from ultralytics import YOLO, solutions
from vidgear.gears import CamGear
import logging

# Configure logging
logging.basicConfig(filename='./logs/tracking_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize CamGear stream
youtube_stream = CamGear(
    source="https://www.youtube.com/live/OIqUka8BOS8?si=DVQmFImFtmlBB4QR",
    stream_mode=True,
    logging=True
).start()

# Read the first frame to get the video dimensions
frame = youtube_stream.read()
if frame is None:
    print("Error: Unable to read the first frame.")
    exit()

height, width, _ = frame.shape

# Calculate the region points based on the video dimensions
region_points = [
    (int(width * 0.05), int(height * 0.9)),  # top-left
    (int(width * 0.95), int(height * 0.9)),  # top-right
    (int(width * 0.95), int(height * 0.1)),  # bottom-right
    (int(width * 0.05), int(height * 0.1)),  # bottom-left
]

# Initialize YOLO model
model = YOLO("./models/yolov8n.pt")

# Initialize Object Counter
counter = solutions.ObjectCounter(
    names=model.names, # Dictionary of classes names
    reg_pts=region_points, # List of points defining the counting region.
    count_reg_color=(0, 255, 0), # RGB color of the counting region.
    count_txt_color=(255, 255, 255), # RGB color of the count text.
    count_bg_color=(0, 0, 0), # RGB color of the count text background.
    line_thickness=2, # Line thickness for bounding boxes.
    track_thickness=2, # Thickness of the track lines.
    view_img=True, # Flag to control whether to display the video stream.
    view_in_counts=True, # Flag to control whether to display the in counts on the video stream.
    view_out_counts=True, # Flag to control whether to display the out counts on the video stream.
    draw_tracks=True, # Flag to control whether to draw the object tracks.
    track_color=(0, 255, 0), # RGB color of the tracks.
    region_thickness=5, # Thickness of the object counting region.
    line_dist_thresh=15, # Euclidean distance threshold for line counter.
    cls_txtdisplay_gap=50, # Display gap between each class count.
)

while True:
    frame = youtube_stream.read()
    if frame is None:
        break

    tracker_config_path = os.path.join('utils', 'bytetrack.yaml')

    # Track objects in the frame
    results = model.track(
        source=frame, # source directory for images or videos
        persist=True, # persisting tracks between frames
        tracker=tracker_config_path, # Tracking method 'bytetrack' or 'botsort'
        conf=0.5, # Confidence Threshold
        iou=0.5, # IOU Threshold
        classes=14, # filter results by class, i.e. classes=0, or classes=[0,2,3]
        verbose=True # Display the object tracking results
    )

    # Use the counter to start counting objects in the frame
    annotated_frame = counter.start_counting(frame, results)

    for result in results:
        
        # PRINT SPEED
        print("Speed Statistics:")
        print("Preprocess: {:.3f}".format(result.speed['preprocess']))
        print("Inference: {:.3f}".format(result.speed['inference']))
        print("Postprocess: {:.3f}".format(result.speed['postprocess']))
        print()

        print("Frame Statistics:")
        print("  Number of boxes:", len(result.boxes))
        confidences = [box.conf for box in result.boxes]
        # print("  Average confidence:", sum(confidences) / len(confidences))
        print("  Number of unique IDs:", len(set(box.id for box in result.boxes)))
        unique_ids = list(set(box.id for box in result.boxes))
        unique_classes = [result.names[int(box.cls)] for box in result.boxes]
        print("  Unique IDs and their classes:")
        for id in unique_ids:
            try:
                cls = result.names[int(next(box.cls for box in result.boxes if box.id == id))]
                print(f"    ID {id.item():.0f}, Class: {cls}")
            except AttributeError:
                print(f"    ID {id}, Class: Unknown")
        print()

        print("Individual Objects:")
        for i, box in enumerate(result.boxes):
            print(f"  Object {i+1}:")
            print(f"    Location: ({box.xyxy[0][0]:.5f}, {box.xyxy[0][1]:.5f}, {box.xyxy[0][2]:.5f}, {box.xyxy[0][3]:.5f})")
            print(f"    Class ID: {box.cls[0].item():.0f}")
            print(f"    Class Name: {result.names[int(box.cls[0].item())]}")
            print(f"    Confidence: {box.conf[0].item():.4f}")
            try:
                print(f"    Tracking ID: {box.id[0].item():.0f}")
            except TypeError:
                print("    Tracking ID: Unknown")
            print()





        # print("########## Result attributes:")
        # for attr in dir(result):
        #     print(f"  {attr}: {getattr(result, attr)}")

        # print("\n########## Result boxes attributes:")
        # if hasattr(result, 'boxes'):
        #     for attr in dir(result.boxes):
        #         print(f"  {attr}: {getattr(result.boxes, attr)}")

        # print("\n########## Result masks attributes:")
        # if hasattr(result, 'masks'):
        #     for attr in dir(result.masks):
        #         print(f"  {attr}: {getattr(result.masks, attr)}")

        # print("\n########## Result probs attributes:")
        # if hasattr(result, 'probs'):
        #     for attr in dir(result.probs):
        #         print(f"  {attr}: {getattr(result.probs, attr)}")

        # print("\n########## Result keypoints attributes:")
        # if hasattr(result, 'keypoints'):
        #     for attr in dir(result.keypoints):
        #         print(f"  {attr}: {getattr(result.keypoints, attr)}")

        # print("\n########## Result speed attributes:")
        # if hasattr(result, 'speed'):
        #     for attr, value in result.speed.items():
        #         print(f"  {attr}: {value}")

        # print("\n########## Result names attributes:")
        # if hasattr(result, 'names'):
        #     for attr, value in result.names.items():
        #         print(f"  {attr}: {value}")

        # print("\n########## Result path:")
        # if hasattr(result, 'path'):
        #     print(f"  path: {result.path}")



    # # Iterate over each Results object in the results_list
    # for results in results_list:
    #     if results.boxes:
    #         for box in results.boxes:
    #             xyxy = box.xyxy[0].cpu().numpy().astype(int)  # Get the bounding box coordinates
    #             conf = box.conf[0].item()  # Get the confidence
    #             class_id = int(box.cls[0])  # Get the class ID
    #             class_name = model.names[class_id]  # Get the class name

    #             # Check if tracking ID is available
    #             if box.id is not None:
    #                 track_id = int(box.id[0].item())  # Get the tracking ID
    #             else:
    #                 track_id = 'N/A'  # Default value if tracking ID is not available

    #             # Draw bounding box
    #             cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

    #             # Prepare the label
    #             label = f"ID: {track_id} {class_name} {conf:.2f}"
    #             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    #             label_background = (xyxy[0], xyxy[1] - h - 10)

    #             # Draw background rectangle for the label
    #             cv2.rectangle(annotated_frame, label_background, (xyxy[0] + w, xyxy[1]), (0, 255, 0), -1)

    #             # Put text on the frame
    #             cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the frame with bounding boxes and counter
    cv2.imshow("Annotated Frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






    # # Use the counter to start counting objects in the frame
    # annotated_frame = counter.start_counting(frame, tracks)

    # # Display the frame with bounding boxes and counter
    # cv2.imshow("Annotated Frame", annotated_frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

youtube_stream.stop()
cv2.destroyAllWindows()






    # # GET KEYS
    # for results in results_list:
    #     # Print the attributes and methods of the results object
    #     print("Results object attributes and methods:")
    #     print(dir(results))

    #     # Print the results content to understand its structure
    #     print("Results content:")
    #     print(results)
        
    #     # Optional: Print detailed attributes of boxes
    #     if results.boxes:
    #         print("Results boxes attributes and methods:")
    #         print(dir(results.boxes))
    #         print("Results boxes content:")
    #         print(results.boxes)
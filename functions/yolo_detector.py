# functions/yolo_detector.py
import logging
from collections import defaultdict

class YoloDetector:
    def __init__(self, model):
        self.model = model
        self.detections = []

        # Configure logging to save logs to a file and append to it
        logging.basicConfig(filename='./logs/yolo_detector.log', level=logging.DEBUG, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                            filemode='a')  # Ensure appending mode
        self.logger = logging.getLogger(__name__)
        self.logger.info("YoloDetector initialized.")

    def detect_and_track(self, frame):
        results = self.model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
        
        annotated_frame = frame.copy()
        for result in results:
            for det in result.boxes.data.tolist():
                if len(det) >= 7:  # Check if det has at least 7 elements
                    detection = {
                        'class': result.names[int(det[6])],  # Use det[6] as the class ID
                        'id': int(det[4]),  # Use det[4] as the detection ID
                        'confidence': det[5]  # Use det[5] as the confidence score
                    }
                else:
                    # Handle the case where det[6] is not present
                    detection = {
                        'class': 'Unknown',  # Default to 'Unknown' class
                        'id': int(det[4]),  # Use det[4] as the detection ID
                        'confidence': det[5]  # Use det[5] as the confidence score
                    }
                self.detections.append(detection)
                ### LOG ###
                self.logger.info(f"Detection: {detection['class']}, ID: {detection['id']}, Confidence: {detection['confidence']:.2f}")
                print(f"Detection: {detection['class']}, ID: {detection['id']}, Confidence: {detection['confidence']:.2f}")
            annotated_frame = result.plot()
        
        return annotated_frame
    
    def summarize_detections(self):
        if not self.detections:
            ### LOG ###
            self.logger.info("No detections to summarize.")
            print("No detections to summarize.")
            return

        summary = defaultdict(lambda: defaultdict(list))

        for detection in self.detections:
            class_name = detection['class']
            det_id = detection['id']
            confidence = detection['confidence']
            summary[class_name][det_id].append(confidence)

        print("Summary of detections:")
        for class_name, ids in summary.items():
            for det_id, confidences in ids.items():
                avg_confidence = sum(confidences) / len(confidences)
                ### LOG ###
                self.logger.info(f"Detection: {class_name}, ID: {det_id}, Average Confidence: {avg_confidence:.2f}")
                print(f"Detection: {class_name}, ID: {det_id}, Average Confidence: {avg_confidence:.2f}")








# # FUNCTIONG INDIVIDUAL DETECTION WRITER
# class YoloDetector:
#     def __init__(self, model):
#         self.model = model
#         self.detections = []

#     def detect_and_track(self, frame):
#         results = self.model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
        
#         self.detections = []  # Clear previous detections
#         annotated_frame = frame.copy()
#         for result in results:
#             # print(result.names)
#             for det in result.boxes.data.tolist():
#                 if len(det) >= 7:  # Check if det has at least 7 elements
#                     detection = {
#                         'class': result.names[int(det[6])],  # Use det[6] as the class ID
#                         'id': int(det[4]),  # Use det[4] as the detection ID
#                         'confidence': det[5]  # Use det[5] as the confidence score
#                     }
#                 else:
#                     # Handle the case where det[6] is not present
#                     detection = {
#                         'class': 'Unknown',  # Default to 'Unknown' class
#                         'id': int(det[4]),  # Use det[4] as the detection ID
#                         'confidence': det[5]  # Use det[5] as the confidence score
#                     }
#                 self.detections.append(detection)
#                 print(f"Detection: {detection['class']}, ID: {detection['id']}, Confidence: {detection['confidence']:.2f}")
#             annotated_frame = result.plot()
        
#         return annotated_frame

















    # def get_summary(self):
    #     class_dict = {}
    #     for det in self.detections:
    #         class_name = det['class']
    #         if class_name not in class_dict:
    #             class_dict[class_name] = {'ids': [], 'count': 0}
    #         class_dict[class_name]['ids'].append(det['id'])
    #         class_dict[class_name]['count'] += 1
        
    #     summary = []
    #     for class_name, info in class_dict.items():
    #         ids = ', '.join(map(str, set(info['ids'])))  # Remove duplicate IDs
    #         summary.append(f"Class: {class_name}, IDs: {ids}, Total Class Detections: {info['count']}")
        
    #     return ' | '.join(summary)













# FUNCTIONING W/O CLASS NAMES & IDS
# class YoloDetector:
#     def __init__(self, model):
#         self.model = model
        
#     def detect_and_track(self, frame):
#         results = self.model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
        
#         annotated_frame = frame.copy()
#         for result in results:
#             annotated_frame = result.plot()
        
#         return annotated_frame

# functions/yolo_detector.py
class YoloDetector:
    def __init__(self, model):
        self.model = model
        self.detections = []

    def detect_and_track(self, frame):
        results = self.model.track(source=frame, show=False, persist=True, tracker="./utils/bytetrack.yaml")
        
        self.detections = []  # Clear previous detections
        annotated_frame = frame.copy()
        for result in results:
            print(result.names)
            for det in result.boxes.data.tolist():
                print("det:", det)
                print("det[4]:", det[4])
                print("det[5]:", det[5])
                print("det[6]:", det[6])
                print("---")
                detection = {
                    'class': result.names[int(det[6])],  # Use det[6] as the class ID
                    'id': int(det[4]),  # Use det[4] as the detection ID
                    'confidence': det[5]  # Use det[5] as the confidence score
                }
                self.detections.append(detection)
                print(f"Detection: {detection['class']}, ID: {detection['id']}, Confidence: {detection['confidence']:.2f}")
            annotated_frame = result.plot()
        
        return annotated_frame

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

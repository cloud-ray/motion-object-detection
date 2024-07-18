from ultralytics import YOLO

model = YOLO("../models/yolov10n.pt")

print(model.model.names)  # print the class names (e.g., object detection class labels)
# print(model.model.nc)  # print the number of classes (num_classes)
# print(dir(model))  # prints all attributes and methods of the outer model
# print(dir(model.model))  # prints all attributes and methods of the inner model
# print(model.model.args)  # print the arguments (hyperparameters) of the inner model

# imgsz = model.model.args['imgsz']
# print(imgsz)  # Output: 640
from ultralytics import YOLO

# Load the YOLOv11n model
model = YOLO('yolo11n.pt')

# Replace 'path/to/your/image.jpg' with the actual path to your image
image_path = 'bus.jpg'

# Predict on the image
results = model(image_path)

# Process the results (optional - for example, print bounding boxes)
for result in results:
    boxes = result.boxes
    print(boxes)

# The processed image with detections will be saved in a 'runs/predict/' directory
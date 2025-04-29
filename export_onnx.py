from ultralytics import YOLO

# Load your YOLO model
model = YOLO('yolo11n.pt')  # Replace 'xxx.pt' with the actual path to your model file

# Export the model to ONNX format
model.export(format='onnx')

print("Model exported to ONNX successfully!")
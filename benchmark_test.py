from ultralytics import YOLO
import time
import torch



# Load the ONNX model using Ultralytics wrapper
model = YOLO("yolo11n.onnx")
# model = YOLO("yolo11n.pt")

# Simulate input image (change to your real image size if needed)
input_tensor = torch.randn(1, 3, 640, 640)

# Warm-up (important for accurate timing)
for _ in range(10):
    _ = model(input_tensor)

# Timing setup
num_runs = 100
start = time.time()

for _ in range(num_runs):
    _ = model(input_tensor)

end = time.time()

# Compute latency and FPS
total_time = end - start
avg_latency_ms = (total_time / num_runs) * 1000
fps = 1000 / avg_latency_ms

print(f"Average Latency: {avg_latency_ms:.2f} ms")
print(f"FPS: {fps:.2f}")

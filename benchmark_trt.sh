#!/bin/bash
 

echo "test for trt model" 

/home/robot/TensorRT-8.6.1.6/bin/trtexec  --loadEngine=yolo11n.trt 
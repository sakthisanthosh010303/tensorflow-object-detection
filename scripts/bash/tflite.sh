#!/bin/bash
#
# Author: Sakthi Santhosh
# Created on: 04/09/2022
#
# Tensorflow Object Detection - Tensorflow Lite Model Converter
tflite_convert \
  --saved_model_dir=./workspace/export/tflite/saved_model/ \
  --output_file=./workspace/export/tflite/saved_model/custom.tflite \
  --input_shapes=1,300,300,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
  --inference_type=FLOAT --allow_custom_ops

#!/bin/bash
#
# Author: Sakthi Santhosh
# Created on: 04/09/2022
#
# Tensorflow Object Detection - Model Freezer
python3 ./tools/research/object_detection/export_tflite_graph_tf2.py \
  --pipeline_config_path=./models/custom/pipeline.config \
  --trained_checkpoint_dir=./models/custom/ \
  --output_directory=./workspace/export/tflite/

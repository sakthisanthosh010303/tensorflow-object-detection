#!/bin/bash
#
# Author: Sakthi Santhosh
# Created on: 01/09/2022
#
# Tensorflow Object Detection - Model Trainer
python3 ./tools/research/object_detection/model_main_tf2.py \
  --model_dir=./models/custom/ \
  --pipeline_config_path=./models/custom/pipeline.config \
  --num_train_steps=2000

#!/bin/bash

MODEL=$1
INPUT_SHAPE=$2
OUTPUT_PATH=$3
MODEL_NAME=$4
IO16=$5

mo --input_model $MODEL --model_name ${MODEL_NAME}_fp32 --input_shape $INPUT_SHAPE --output_dir $OUTPUT_PATH
mo --input_model $MODEL --model_name ${MODEL_NAME}_fp16 --input_shape $INPUT_SHAPE --output_dir $OUTPUT_PATH --compress_to_fp16

if [ ${IO16} = "True" ]; then
mo --input_model $MODEL --model_name ${MODEL_NAME}_fp32_io16 --input_shape $INPUT_SHAPE --output_dir $OUTPUT_PATH --data_type FP16
mo --input_model $MODEL --model_name ${MODEL_NAME}_fp16_io16 --input_shape $INPUT_SHAPE --output_dir $OUTPUT_PATH --data_type FP16 --compress_to_fp16
fi
#!/bin/bash

MODEL=$1
MIN_SHAPE=$2
OPT_SHAPE=$3
MAX_SHAPE=$4
OUTPUT_PATH=$5
IO16=$6

trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32.trt --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16.trt --fp16 --minShapes=$MIN_SHAPE --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE

if [ ${IO16} = "True" ]; then
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32_io16.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16_io16.trt --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
fi
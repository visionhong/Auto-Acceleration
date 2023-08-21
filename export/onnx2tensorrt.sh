#!/bin/bash

MODEL=$1
OUTPUT_PATH=$2
MIN_SHAPE=$3
OPT_SHAPE=$4
MAX_SHAPE=$5
IO16=$6
DYNAMIC=$7

if [ ${DYNAMIC} = "False" ]; then
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32.plan 
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16.plan --fp16 

    if [ ${IO16} = "True" ]; then
        trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32_io16.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw 
        trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16_io16.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    fi
else
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32.plan --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
    trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16.plan --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE --fp16

    if [ ${IO16} = "True" ]; then
        trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp32_io16.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
        trtexec --onnx=$MODEL --saveEngine=${OUTPUT_PATH}_fp16_io16.plan --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --minShapes=$MIN_SHAPE --optShapes=$OPT_SHAPE --maxShapes=$MAX_SHAPE
    fi
fi

## Auto Acceleration

**Environment**

* Image: nvcr.io/nvidia/pytorch:22.08-py3
* TensorRT: 8.4.2
* OpenVINO: 2022.3.0
* ONNX: 1.12.0
* onnxruntime-gpu: 1.14.1
* pycuda: 2022.2.2

**INPUT**

* onnx file
* config.yml(device, I/O tensor info)

**OUTPUT**

* An Excel file containing information such as throughput, filesize, and output tensor.
* converted model file.

`<br/>`

## Getting Started

1. Clone this repository

   ```bash
   git clone https://github.com/visionhong/Auto-Acceleration.git
   ```
2. Place the ONNX file in the input/model directory and modify the config.yml file in the input/config directory to match your model
3. Run the convert and inference.

   ```bash
   docker compose up
   ```
4. Check the results in the output folder.


## Notice

* This application was developed based on PyTorch models. Testing has not been conducted on machine learning models or Tensorflow models.
* The current version of the container only supports ONNX opset_version up to 16
* There is a phenomenon where the output tensor of TensorRT becomes nan when converting from a PyTorch model with weights in fp16 mode to ONNX. Therefore, it is recommended to convert a PyTorch model with weights in fp32 mode to ONNX.
* Writing the config.yml file must follow the rules below.

  1. The input name must be the same as the input_names used when converting to ONNX.
  2. The input order must be written in the order required by the model
  3. When converting to ONNX, only set the range of the dimensions that are set as dynamic axes, as the dimensions that are not set as dynamic axes must use fixed values.

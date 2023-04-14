import onnx
from onnxconverter_common import float16

def convert_fp16(model_path, model_name, io16):
    
    model = onnx.load(model_path)
    try:
        model_fp16 = float16.convert_float_to_float16(model)

        if io16:
            name = f"output/onnx/{model_name}_fp16_io16.onnx"
        else:
            name = f"output/onnx/{model_name}_fp16.onnx"

        onnx.save(model_fp16, name)
    
    except ValueError as e:
        print(f"Error: {str(e)}")
    
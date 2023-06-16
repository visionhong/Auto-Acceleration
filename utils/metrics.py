import os
import numpy as np
import pandas as pd
from glob import glob

from utils.onnx_prediction import onnxruntime_infer
from utils.openvino_prediction import openvino_infer
from utils.tensorrt_prediction import tensorrt_infer

def get_result(name, config, iteration, device):

    result = {}
    time = {}
    file_size = {}


    dummy_data = {}
    for key, value in config['input'].items(): 
        dummy_data[key] = np.ones(value['use_shape']).astype(value['dtype'])
    

    for file_path in glob("./output/onnx/*.onnx"):
        file_name = os.path.basename(file_path)
        exp_name = os.path.splitext(file_name)[0].replace(name, "onnx")

        result[exp_name], time[exp_name], file_size[exp_name] = onnxruntime_infer(
                model_path=f'./output/onnx/{file_name}',
                config=config,
                data=dummy_data.copy(),
                io16 = True if 'io16' in file_name else False,
                iteration=iteration,
                device=device
            )
            
    if device == 'cpu':
        for file_name in [i for i in os.listdir('./output/openvino') if i.endswith('.xml')]:
            exp_name = os.path.splitext(file_name)[0].replace(name, "openvino")

            result[exp_name], time[exp_name], file_size[exp_name] = openvino_infer(
                    model_path=f'./output/openvino/{file_name}',
                    config=config,
                    data=dummy_data.copy(),
                    io16 = True if 'io16' in file_name else False,
                    iteration=iteration
                )

    else: 
        for file_name in os.listdir('./output/tensorrt'):
            exp_name = os.path.splitext(file_name)[0].replace(name, "tensorrt")

            result[exp_name], time[exp_name], file_size[exp_name] = tensorrt_infer(
                model_path=f'./output/tensorrt/{file_name}',
                config=config,
                data=dummy_data.copy(),
                io16 = True if 'io16' in file_name else False,
                iteration=iteration
            )


    return result, time, file_size



def mae(output1, output2):
    return np.mean(np.abs(output1 - output2))


def get_metrics(name, config):

    iteration = 50
    result, time, file_size = get_result(name, config, iteration, config['device'])
    
    summary = []
    standard = np.array(result['onnx'])
    
    for key, value in result.items():
        summary.append([
                        key, 
                        f'{round(file_size[key], 2)}', 
                        f'{round(iteration/time[key], 2)}',
                        mae(standard, np.array(value))])

    df = pd.DataFrame(columns=['task', 'file size(MB)', 'throughput(samples/second)', 'mae'], data=summary)
    
    df.to_excel('./output/summary.xlsx', index=False)
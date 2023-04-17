import os
import subprocess, shlex
import shutil
from glob import glob

from onnxruntime import InferenceSession
from export.onnx_convert import convert_fp16
from utils.metrics import get_metrics

import yaml


def parse_data_info(onnx_path):
    with open('./input/config/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['input'] = {str(key): value for key, value in config['input'].items()} # name이 int인 경우 str로 변환
    config['output'] = {str(key): value for key, value in config['output'].items()}
    session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    for i in session.get_inputs():
        config['input'][i.name]['dtype'] = i.type[7:-1]+'32' if i.type[7:-1] == 'float' else i.type[7:-1]
        
    for o in session.get_outputs():
        config['output'][o.name]['dtype'] = o.type[7:-1]+'32' if o.type[7:-1] == 'float' else o.type[7:-1]

    return config


def main():
    os.makedirs('./output/onnx', exist_ok=True)    

    for i in os.listdir('./input/model'):
        shutil.copy2(f'./input/model/{i}', f'./output/onnx/{i}')

    onnx_path = glob("./output/onnx/*.onnx")[0]
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    config = parse_data_info(onnx_path)    

    io16 = True
    for i in config['input'].values():
        if 'int' in i['dtype']:

            break

    if config['device'] == 'cpu':
        input_shape = ''
        for i in config['input']:
            name = config['input'][i]

            shape = [value if name['min_shape'][idx] == name['max_shape'][idx] else -1 for idx, value in enumerate(name['use_shape'])]
            input_shape += f"{shape},"


        subprocess.run(["chmod", "+x", "export/onnx2openvino.sh"])
        subprocess.call(shlex.split(f"export/onnx2openvino.sh {onnx_path} {''.join(input_shape[:-1].split())} output/openvino {model_name} {io16}"))

    
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])

        min_shape, max_shape, opt_shape = '', '', ''

        for i in config['input']:
            min_shape += (i + ':' + "x".join(map(str, config['input'][i]['min_shape'])) + ',')  
            max_shape += (i + ':' + "x".join(map(str, config['input'][i]['max_shape'])) + ',')  
            opt_shape += (i + ':' + "x".join(map(str, config['input'][i]['use_shape'])) + ',')  

        convert_fp16(onnx_path, model_name, io16)
        os.makedirs('./output/tensorrt', exist_ok=True)
        
        subprocess.run(["chmod", "+x", "export/onnx2tensorrt.sh"])
        subprocess.call(shlex.split(f"export/onnx2tensorrt.sh {onnx_path} {min_shape[:-1]} {opt_shape[:-1]} {max_shape[:-1]} output/tensorrt/{model_name} {io16}"))


    
    # ================== metric ====================
    
    get_metrics(model_name, config)



if __name__ == '__main__':
    main()
import os
import numpy as np
import onnxruntime as ort
from time import time
from tqdm import tqdm
from utils.folder_size import get_dir_size


def onnxruntime_infer(model_path, config, data, io16, iteration, device):

    if device == 'cpu':
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    else:
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    inf_time = 0
    
    if io16:
        for key in data:
            data[key] = data[key].astype(np.float16)

    # warm-up
    for _ in range(3):

        session.run(list(config['output'].keys()), input_feed=data)
        
    for _ in tqdm(range(iteration)): 
        
        # 모델을 실행합니다.
        start_time = time()
        output = session.run(list(config['output'].keys()), input_feed=data)
        end_time = time()

        inf_time += (end_time - start_time)

    result = [i.flatten()[0] for i in output]  # output의 첫번째 텐서만 가져옴

    if len(os.listdir(os.path.dirname(model_path))) > 5:  # onnx 파일이 2GB가 넘어서 분할된 경우 
        size = get_dir_size(os.path.dirname(model_path)) / (1024 * 1024)
    else:
        size = os.stat(model_path).st_size / (1024 * 1024)
        
    return result, inf_time, size
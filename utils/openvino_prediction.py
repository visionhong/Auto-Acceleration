import os
import numpy as np

from openvino.runtime import Core
from time import time
from tqdm import tqdm


def openvino_infer(model_path, config, data, io16, iteration):

    # Initialize inference engine core
    ie = Core()

    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    
    inf_time = 0
    
    if io16:
        for key in data:
            data[key] = data[key].astype(np.float16)

    data = tuple(data.values())

    # warm_up
    for _ in range(3):
        compiled_model(data)

    for _ in tqdm(range(iteration)): 
                
        start_time = time()
        output = compiled_model(data)
        end_time = time()

        inf_time += (end_time - start_time)
    
    result = [output[compiled_model.outputs[i]] for i in range(len(config['output']))]
    
    return result, inf_time, os.stat(os.path.splitext(model_path)[0]+'.bin').st_size / (1024 * 1024)
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from time import time
from tqdm import tqdm


def tensorrt_infer(model_path, config, data, io16, iteration):
    
    def predict(): # result gets copied into output

        for key in data.keys():
            cuda.memcpy_htod_async(d_input[key], data[key], stream)

        # execute model
        context.execute_async_v2(bindings, stream.handle)
        # transfer predictions back
        for key in output.keys():
            cuda.memcpy_dtoh_async(output[key], d_output[key], stream)

        # syncronize threads
        stream.synchronize()
        return output
    
    trt.init_libnvinfer_plugins(None,'')
    with open(model_path, 'rb') as f:
        serialized_engine = f.read()

    # Deserialize the engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # Create execution context
    context = engine.create_execution_context()

    inf_time = 0       
    
    if io16:
        for key in data:
            data[key] = data[key].astype(np.float16)

    output = {}
    for key, value in config['output'].items(): 
        output[key] = np.zeros(value['use_shape']).astype(value['dtype'] if not io16 else np.float16)

    d_input = {key:cuda.mem_alloc(1 * values.nbytes) for key, values in data.items()}
    d_output = {key:cuda.mem_alloc(1 * values.nbytes) for key, values in output.items()}
    
    bindings = [int(i) for i in d_input.values()] + [int(i) for i in d_output.values()]
    

    stream = cuda.Stream()

    for _ in tqdm(range(iteration)):
        
        start_time = time()
        preds = predict()
        end_time = time()

        inf_time += (end_time - start_time)
        

    result = [i.flatten()[0] for i in preds.values()]

    return result, inf_time, os.stat(model_path).st_size / (1024 * 1024)

import GPUtil

def get_gpu_temperature():
    try:
        gpu = GPUtil.getGPUs()[0]
        return gpu.temperature
    except:
        return 0
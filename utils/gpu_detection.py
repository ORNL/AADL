##########################################

import torch

if torch.cuda.is_available():
    import pycuda
    from pycuda import compiler
    import pycuda.driver as drv

    drv.init()


################################################################

def count_gpus():
    number = 0
    if torch.cuda.is_available():
        number = torch.cuda.device_count()
        print(number, " - GPUs found")
    else:
        print("GPU NOT found")
    return number


def get_gpus_list():
    gpu_list = []
    if torch.cuda.is_available():
        # print("%d device(s) found." % drv.Device.count())
        for ordinal in range(drv.Device.count()):
            dev = drv.Device(ordinal)
            # print (ordinal, dev.name())
            gpu_list.append(ordinal)
    return gpu_list


def get_gpu(number):
    gpu_list = get_gpus_list()
    if torch.cuda.is_available() and (number in gpu_list):
        device = torch.device("cuda:" + str(number))  # you can continue going on here, like cuda:1 cuda:2....etc.
    else:
        device = torch.device("cpu")
        print(" running on the CPU - GPU" + str(number)+" is NOT available")
    return device


#############################################################################

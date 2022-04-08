import logging
import time
import os

import pynvml
import psutil




logger = logging.getLogger('monitor')
logger.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s: %(message)s")
os.makedirs('./monitorLog',exist_ok=True)
fileHandler = logging.FileHandler(os.path.join('./monitorLog', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "_monitor.log"))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
commandHandler = logging.StreamHandler()
commandHandler.setLevel(logging.INFO)
commandHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(commandHandler)

while True:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    logger.info( f'{str(psutil.virtual_memory().used / (2**30))}   {str( meminfo.used/1024**2)}' )
    time.sleep(4)
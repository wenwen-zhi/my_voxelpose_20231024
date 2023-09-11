import logging
import os
import time
logging.root.setLevel(level=logging.INFO)
def create_logger(name,log_dir,level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    os.makedirs(log_dir,exist_ok=True)
    handler = logging.FileHandler(log_dir+f"/log-{time.strftime('%Y%m%d',time.localtime())}.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
def create_app_logger(name,log_subdir_name="main",level=logging.INFO):
    return create_logger(name,os.path.join(os.getcwd(),"log",log_subdir_name),level=level)

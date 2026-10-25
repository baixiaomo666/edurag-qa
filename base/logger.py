import logging
import os
from base.config import config

# 项目根目录（绝对路径，保证日志位置固定）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_file=config.LOG_FILE):
    # 拼接日志绝对路径 → 强制写入项目根目录，不受执行位置影响
    log_path = os.path.join(BASE_DIR, log_file)
    # 自动创建日志文件夹
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # 获取全局日志实例
    logger = logging.getLogger("KaoyanQA")
    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式定义
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件输出处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 控制台输出处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 全局单例日志对象
logger = setup_logging()
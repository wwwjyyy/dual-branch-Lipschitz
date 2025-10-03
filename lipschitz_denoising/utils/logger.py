import logging
import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml

def setup_logger(log_dir, name=None, level=logging.INFO):
    """
    设置日志记录器，同时输出到文件和终端
    
    参数:
        log_dir (str): 日志文件存储目录
        name (str): 日志器名称，默认为None（根日志器）
        level: 日志级别，默认为INFO
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    
    logger.setLevel(level)
    logger.handlers.clear()  # 清除现有处理器
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录日志系统初始化信息
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    
    return logger

def setup_tensorboard(log_dir, config=None):
    """
    设置TensorBoard记录器
    
    参数:
        log_dir (str): TensorBoard日志存储目录
        config (dict): 实验配置，可选
        
    返回:
        SummaryWriter: TensorBoard记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 如果提供了配置，将其记录到TensorBoard
    if config is not None:
        # 将配置转换为YAML格式的字符串
        config_str = yaml.dump(config, default_flow_style=False)
        writer.add_text("Experiment Configuration", f"```yaml\n{config_str}\n```")
    
    return writer
import logging
from pathlib import Path


def setup_logger(name: str = "project", log_file: str = "project.log", level: int = logging.DEBUG) -> logging.Logger:
    """设置日志记录器。"""
    # 创建日志目录（如果不存在）
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 确保日志文件保存到 log_dir 目录下
    log_file_path = log_dir / log_file

    # 创建日志记录器，避免重复添加处理器
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

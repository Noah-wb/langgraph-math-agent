"""
日志记录工具
"""

import os
import logging
import logging.config
import yaml
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, config_path: str = "config/logging.yaml"):
        """
        初始化日志管理器
        
        Args:
            config_path: 日志配置文件路径
        """
        self.config_path = config_path
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        try:
            # 确保日志目录存在
            os.makedirs("logs", exist_ok=True)
            
            # 加载日志配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logging.config.dictConfig(config)
            self.logger = logging.getLogger('math_agent')
            self.logger.info("日志系统初始化成功")
            
        except Exception as e:
            # 如果配置文件加载失败，使用默认配置
            self._setup_default_logging()
            self.logger.error(f"日志配置加载失败，使用默认配置: {e}")
    
    def _setup_default_logging(self):
        """设置默认日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/math_agent.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger('math_agent')
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            logging.Logger 实例
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(f'math_agent.{name}')
        return self.loggers[name]
    
    def log_function_call(self, func_name: str, args: tuple = None, kwargs: dict = None, result: Any = None):
        """
        记录函数调用
        
        Args:
            func_name: 函数名称
            args: 位置参数
            kwargs: 关键字参数
            result: 返回值
        """
        self.logger.debug(f"调用函数: {func_name}")
        if args:
            self.logger.debug(f"  位置参数: {args}")
        if kwargs:
            self.logger.debug(f"  关键字参数: {kwargs}")
        if result is not None:
            self.logger.debug(f"  返回值: {result}")
    
    def log_error(self, error: Exception, context: str = ""):
        """
        记录错误信息
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        error_msg = f"错误: {type(error).__name__}: {str(error)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_performance(self, operation: str, duration: float, details: str = ""):
        """
        记录性能信息
        
        Args:
            operation: 操作名称
            duration: 耗时（秒）
            details: 详细信息
        """
        msg = f"性能: {operation} 耗时 {duration:.3f}s"
        if details:
            msg += f" - {details}"
        self.logger.info(msg)


# 全局日志管理器实例
logger_manager = LoggerManager()


def log_function(func):
    """函数调用日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logger_manager.get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}", exc_info=True)
            raise
    
    return wrapper


def log_class_methods(cls):
    """类方法日志装饰器"""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_function(attr))
    return cls


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or logger_manager.logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"开始执行: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"执行失败: {self.operation} 耗时 {duration:.3f}s")
            else:
                self.logger.info(f"执行完成: {self.operation} 耗时 {duration:.3f}s")


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger 实例
    """
    return logger_manager.get_logger(name)


# 导出常用的日志记录器
main_logger = logger_manager.get_logger('main')
model_logger = logger_manager.get_logger('model')
agent_logger = logger_manager.get_logger('agent')
history_logger = logger_manager.get_logger('history')


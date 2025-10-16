"""
模型管理器
支持 DeepSeek、GLM、Kimi 等模型的统一调用和切换
"""

import os
import yaml
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from .logger import get_logger, PerformanceTimer
from .model_call_logger import model_call_logger

# 加载环境变量
load_dotenv()


class ModelManager:
    """模型管理器类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化模型管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger('model')
        self.config_path = config_path
        self.config = self._load_config()
        self.current_model = None
        self.current_llm = None
        self.logger.info("模型管理器初始化完成")
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"配置文件加载成功: {self.config_path}")
                return config
        except FileNotFoundError:
            self.logger.error(f"配置文件不存在: {self.config_path}")
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件格式错误: {e}")
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get_available_models(self) -> list:
        """获取可用的模型列表"""
        return list(self.config['models'].keys())
    
    def switch_model(self, model_name: str) -> ChatOpenAI:
        """
        切换模型
        
        Args:
            model_name: 模型名称 (deepseek/glm/kimi)
            
        Returns:
            ChatOpenAI 实例
            
        Raises:
            ValueError: 当模型名称无效时
            KeyError: 当 API Key 未设置时
        """
        with PerformanceTimer(f"切换到模型: {model_name}", self.logger):
            if model_name not in self.config['models']:
                self.logger.error(f"不支持的模型: {model_name}")
                raise ValueError(f"不支持的模型: {model_name}")
            
            self.logger.info(f"开始切换到模型: {model_name}")
            model_config = self.config['models'][model_name]
            api_key_env = model_config['api_key_env']
            
            # 优先从环境变量获取，如果没有则从配置文件获取
            api_key = os.getenv(api_key_env)
            if not api_key and api_key_env in model_config:
                api_key = model_config[api_key_env]
                self.logger.info(f"从配置文件获取 {model_name} API Key")
            
            if not api_key:
                self.logger.error(f"API Key 未设置: {api_key_env}")
                raise KeyError(f"请设置环境变量 {api_key_env} 或在配置文件中配置")
            
            # 创建 ChatOpenAI 实例
            llm = ChatOpenAI(
                model=model_config['name'],
                base_url=model_config['base_url'],
                api_key=api_key,
                temperature=model_config.get('temperature', 0.7),
                max_tokens=model_config.get('max_tokens', 2000),
                streaming=True  # 启用流式输出
            )
            
            self.current_model = model_name
            self.current_llm = llm
            self.logger.info(f"成功切换到模型: {model_name}")
            
            # 记录模型切换日志
            model_call_logger.log_model_switch(
                from_model=self.current_model if hasattr(self, 'current_model') else "none",
                to_model=model_name,
                reason="用户手动切换"
            )
            
            return llm
    
    def get_current_model(self) -> Optional[str]:
        """获取当前模型名称"""
        return self.current_model
    
    def get_current_llm(self) -> Optional[ChatOpenAI]:
        """获取当前 LLM 实例"""
        return self.current_llm
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置信息
        """
        if model_name not in self.config['models']:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return self.config['models'][model_name]
    
    def check_api_keys(self) -> Dict[str, bool]:
        """
        检查所有模型的 API Key 是否已设置
        
        Returns:
            字典，键为模型名，值为是否已设置 API Key
        """
        status = {}
        for model_name, config in self.config['models'].items():
            api_key_env = config['api_key_env']
            status[model_name] = bool(os.getenv(api_key_env))
        
        return status

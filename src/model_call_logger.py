"""
模型调用日志记录器
专门记录智能体与大模型之间的交互详情
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

class ModelCallLogger:
    """模型调用专用日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建专门的模型调用日志文件
        self.call_log_file = self.log_dir / "model_calls.log"
        self.detail_log_file = self.log_dir / "model_details.log"
        
        # 设置日志格式
        self.logger = logging.getLogger("model_calls")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.call_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建详细日志处理器
        detail_handler = logging.FileHandler(self.detail_log_file, encoding='utf-8')
        detail_handler.setLevel(logging.DEBUG)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        detail_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(detail_handler)
    
    def log_model_call_start(self, model_name: str, session_id: str, user_input: str, 
                           model_config: Dict[str, Any]) -> str:
        """记录模型调用开始"""
        call_id = f"call_{int(time.time() * 1000)}"
        
        call_info = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "session_id": session_id,
            "user_input": user_input,
            "model_config": model_config,
            "status": "started"
        }
        
        self.logger.info(f"🚀 模型调用开始 - ID: {call_id}, 模型: {model_name}, 会话: {session_id}")
        self.logger.debug(f"调用详情: {json.dumps(call_info, ensure_ascii=False, indent=2)}")
        
        return call_id
    
    def log_model_call_request(self, call_id: str, request_data: Dict[str, Any]):
        """记录发送给模型的请求数据"""
        self.logger.info(f"📤 发送请求 - 调用ID: {call_id}")
        self.logger.debug(f"请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
    
    def log_model_call_response(self, call_id: str, response_data: Dict[str, Any], 
                               response_time: float):
        """记录模型响应"""
        self.logger.info(f"📥 收到响应 - 调用ID: {call_id}, 耗时: {response_time:.3f}s")
        self.logger.debug(f"响应数据: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
    
    def log_model_call_error(self, call_id: str, error: Exception, error_time: float):
        """记录模型调用错误"""
        self.logger.error(f"❌ 调用失败 - 调用ID: {call_id}, 耗时: {error_time:.3f}s, 错误: {str(error)}")
        self.logger.debug(f"错误详情: {str(error)}")
    
    def log_model_call_complete(self, call_id: str, total_time: float, 
                              tokens_used: Optional[int] = None, 
                              cost: Optional[float] = None):
        """记录模型调用完成"""
        cost_info = ""
        if tokens_used:
            cost_info = f", 消耗tokens: {tokens_used}"
        if cost is not None:
            cost_info += f", 成本: ${cost:.4f}"
        self.logger.info(f"✅ 调用完成 - 调用ID: {call_id}, 总耗时: {total_time:.3f}s{cost_info}")
    
    def log_model_switch(self, from_model: str, to_model: str, reason: str = ""):
        """记录模型切换"""
        self.logger.info(f"🔄 模型切换: {from_model} -> {to_model}")
        if reason:
            self.logger.info(f"切换原因: {reason}")
    
    def log_tool_decision(self, call_id: str, decision: str, reasoning: str = "", 
                         tool_calls: Optional[List[Dict]] = None):
        """记录大模型对工具调用的判断"""
        if decision == "use_tools":
            self.logger.info(f"🔧 决定使用工具 - 调用ID: {call_id}")
            if tool_calls:
                for i, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    self.logger.info(f"  工具 {i}: {tool_name}({tool_args})")
        else:
            self.logger.info(f"💬 直接回复用户 - 调用ID: {call_id}")
        
        if reasoning:
            self.logger.debug(f"判断理由: {reasoning}")
    
    def log_tool_execution_start(self, call_id: str, tool_name: str, tool_args: Dict[str, Any]):
        """记录工具执行开始"""
        self.logger.info(f"⚙️ 开始执行工具 - 调用ID: {call_id}, 工具: {tool_name}")
        self.logger.debug(f"工具参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}")
    
    def log_tool_execution_result(self, call_id: str, tool_name: str, result: Any, 
                                 execution_time: float):
        """记录工具执行结果"""
        self.logger.info(f"✅ 工具执行完成 - 调用ID: {call_id}, 工具: {tool_name}, 耗时: {execution_time:.3f}s")
        self.logger.debug(f"执行结果: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")
    
    def log_tool_execution_error(self, call_id: str, tool_name: str, error: Exception, 
                                execution_time: float):
        """记录工具执行错误"""
        self.logger.error(f"❌ 工具执行失败 - 调用ID: {call_id}, 工具: {tool_name}, 耗时: {execution_time:.3f}s, 错误: {str(error)}")
    
    def log_model_health_check(self, model_name: str, status: str, response_time: float):
        """记录模型健康检查"""
        status_emoji = "✅" if status == "healthy" else "❌"
        self.logger.info(f"{status_emoji} 健康检查 - 模型: {model_name}, 状态: {status}, 响应时间: {response_time:.3f}s")
    
    def get_call_history(self, session_id: Optional[str] = None, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """获取调用历史记录"""
        try:
            if not self.call_log_file.exists():
                return []
            
            calls = []
            with open(self.call_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '模型调用开始' in line or '调用完成' in line:
                        # 解析日志行提取关键信息
                        parts = line.split(' - ')
                        if len(parts) >= 2:
                            timestamp = parts[0].split(']')[0].replace('[', '').replace('INFO', '').strip()
                            info = parts[1].strip()
                            
                            call_info = {
                                "timestamp": timestamp,
                                "info": info
                            }
                            calls.append(call_info)
            
            # 如果指定了会话ID，过滤相关记录
            if session_id:
                calls = [call for call in calls if session_id in call.get("info", "")]
            
            return calls[-limit:]  # 返回最新的记录
            
        except Exception as e:
            self.logger.error(f"获取调用历史失败: {str(e)}")
            return []
    
    def clear_logs(self):
        """清空日志文件"""
        try:
            if self.call_log_file.exists():
                self.call_log_file.unlink()
            if self.detail_log_file.exists():
                self.detail_log_file.unlink()
            self.logger.info("日志文件已清空")
        except Exception as e:
            self.logger.error(f"清空日志失败: {str(e)}")

# 全局模型调用日志记录器实例
model_call_logger = ModelCallLogger()

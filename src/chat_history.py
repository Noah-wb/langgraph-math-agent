"""
会话历史管理
使用 JSON 文件存储和管理聊天历史记录
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .logger import get_logger, PerformanceTimer


class ChatHistoryManager:
    """会话历史管理器"""
    
    def __init__(self, history_dir: str = "./chat_history"):
        """
        初始化历史管理器
        
        Args:
            history_dir: 历史文件存储目录
        """
        self.logger = get_logger('history')
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.current_session_id = None
        self.current_history = []
        self.logger.info(f"历史管理器初始化完成，存储目录: {history_dir}")
    
    def _get_session_file(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return self.history_dir / f"session_{session_id}.json"
    
    def create_session(self) -> str:
        """
        创建新的会话
        
        Returns:
            会话ID
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_id
        self.current_history = []
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """
        加载指定会话的历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否加载成功
        """
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.current_session_id = session_id
                self.current_history = data.get('messages', [])
            return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def save_session(self) -> bool:
        """
        保存当前会话到文件
        
        Returns:
            是否保存成功
        """
        if not self.current_session_id:
            return False
        
        session_file = self._get_session_file(self.current_session_id)
        
        try:
            data = {
                'session_id': self.current_session_id,
                'created_at': datetime.now().isoformat(),
                'messages': self.current_history
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加消息到当前会话
        
        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 额外的元数据
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            message['metadata'] = metadata
        
        self.current_history.append(message)
    
    def get_current_history(self) -> List[Dict[str, Any]]:
        """获取当前会话的历史记录"""
        return self.current_history.copy()
    
    def get_session_list(self) -> List[Dict[str, Any]]:
        """
        获取所有会话列表
        
        Returns:
            会话信息列表
        """
        sessions = []
        
        for session_file in self.history_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id', ''),
                        'created_at': data.get('created_at', ''),
                        'message_count': len(data.get('messages', []))
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # 按创建时间倒序排列
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions
    
    def clear_current_session(self):
        """清除当前会话历史"""
        self.current_history = []
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        session_file = self._get_session_file(session_id)
        
        if session_file.exists():
            try:
                session_file.unlink()
                return True
            except Exception:
                return False
        
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典，如果不存在则返回 None
        """
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'session_id': data.get('session_id', ''),
                    'created_at': data.get('created_at', ''),
                    'message_count': len(data.get('messages', []))
                }
        except (json.JSONDecodeError, KeyError):
            return None

"""
LangGraph 数学智能体 CLI 主程序
提供交互式命令行界面，支持流式对话、模型切换和会话管理
"""

import sys
import os
import yaml
from typing import Optional

# 添加 src 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import ModelManager
from src.chat_history import ChatHistoryManager
from src.agent import MathAgent
from src.logger import get_logger, PerformanceTimer


class MathAgentCLI:
    """数学智能体命令行界面"""
    
    def __init__(self):
        """初始化 CLI"""
        self.logger = get_logger('main')
        self.model_manager = ModelManager()
        self.history_manager = ChatHistoryManager()
        self.agent = None
        self.current_session_id = None
        self.config = self._load_config()
        self.logger.info("CLI 初始化完成")
        
    def print_banner(self):
        """打印欢迎横幅"""
        print("=" * 60)
        print("🤖 LangGraph 智能助手")
        print("=" * 60)
        print("支持模型: DeepSeek | GLM | Kimi")
        print("数据工具: CSV文件分析、统计分析、数据筛选")
        print("支持功能: 流式对话 | 模型切换 | 会话历史")
        print("=" * 60)
        print()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open('config/config.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ 配置文件加载失败: {e}")
            return {}
    
    def select_model(self) -> str:
        """选择模型"""
        # 检查配置是否跳过模型选择
        if self.config.get('app', {}).get('auto_use_default', False):
            default_model = self.config.get('default_model', 'deepseek')
            print(f"🚀 自动使用默认模型: {default_model}")
            return default_model
        
        print("请选择要使用的模型:")
        print("1. DeepSeek")
        print("2. GLM (智谱AI)")
        print("3. Kimi (月之暗面)")
        print()
        
        # 检查 API Keys
        api_status = self.model_manager.check_api_keys()
        available_models = []
        
        for i, (model_name, has_key) in enumerate(api_status.items(), 1):
            status = "✅" if has_key else "❌"
            print(f"{i}. {model_name.upper()} {status}")
            if has_key:
                available_models.append(model_name)
        
        print()
        
        while True:
            try:
                choice = input("请输入选择 (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    model_map = {'1': 'deepseek', '2': 'glm', '3': 'kimi'}
                    selected_model = model_map[choice]
                    
                    if selected_model in available_models:
                        return selected_model
                    else:
                        print(f"❌ {selected_model.upper()} 的 API Key 未设置，请检查 .env 文件")
                        continue
                else:
                    print("❌ 请输入 1、2 或 3")
            except KeyboardInterrupt:
                print("\n👋 再见！")
                sys.exit(0)
    
    def initialize_agent(self, model_name: str):
        """初始化智能体"""
        try:
            print(f"🔄 正在初始化 {model_name.upper()} 模型...")
            self.model_manager.switch_model(model_name)
            self.agent = MathAgent(self.model_manager, self.history_manager)
            self.current_session_id = self.history_manager.create_session()
            print(f"✅ {model_name.upper()} 模型初始化成功！")
            print()
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            sys.exit(1)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n📖 可用命令:")
        print("  /switch <model>  - 切换模型 (deepseek/glm/kimi)")
        print("  /history        - 查看会话历史")
        print("  /clear          - 清除当前会话历史")
        print("  /help           - 显示此帮助信息")
        print("  /exit 或 /quit  - 退出程序")
        print("\n🔧 支持功能:")
        print("  • 数学计算: 加减乘除运算")
        print("  • 数据分析: CSV文件分析、统计、筛选、聚合")
        print("  • 流式对话: 实时响应和工具调用")
        print("  • 会话管理: 保存和恢复对话历史")
        print()
    
    def handle_command(self, command: str) -> bool:
        """
        处理特殊命令
        
        Args:
            command: 命令字符串
            
        Returns:
            是否应该继续对话
        """
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            self.print_help()
            return True
        
        elif cmd == "/switch":
            if len(parts) < 2:
                print("❌ 用法: /switch <model>")
                print("可用模型: deepseek, glm, kimi")
                return True
            
            model_name = parts[1].lower()
            if model_name not in ['deepseek', 'glm', 'kimi']:
                print("❌ 不支持的模型，可用模型: deepseek, glm, kimi")
                return True
            
            try:
                self.model_manager.switch_model(model_name)
                self.agent = MathAgent(self.model_manager, self.history_manager)
                print(f"✅ 已切换到 {model_name.upper()} 模型")
            except Exception as e:
                print(f"❌ 切换模型失败: {e}")
            return True
        
        elif cmd == "/history":
            sessions = self.history_manager.get_session_list()
            if not sessions:
                print("📝 暂无会话历史")
            else:
                print("📝 会话历史:")
                for i, session in enumerate(sessions[:5], 1):  # 显示最近5个会话
                    print(f"  {i}. {session['session_id']} ({session['message_count']} 条消息)")
            return True
        
        elif cmd == "/clear":
            self.history_manager.clear_current_session()
            self.current_session_id = self.history_manager.create_session()
            print("🗑️ 当前会话历史已清除")
            return True
        
        elif cmd in ["/exit", "/quit"]:
            print("👋 再见！")
            return False
        
        else:
            print(f"❌ 未知命令: {cmd}")
            print("输入 /help 查看可用命令")
            return True
    
    def chat_loop(self):
        """主对话循环"""
        print("💬 开始对话吧！输入 /help 查看帮助，输入 /exit 退出")
        print("-" * 60)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # 与智能体对话
                print("\n🤖 智能体: ", end="", flush=True)
                
                # 流式输出响应
                response_parts = []
                for chunk in self.agent.chat_stream(user_input, self.current_session_id):
                    print(chunk, end="", flush=True)
                    response_parts.append(chunk)
                
                print()  # 换行
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("请重试或输入 /help 查看帮助")
    
    def run(self):
        """运行 CLI"""
        self.print_banner()
        
        # 选择模型
        model_name = self.select_model()
        
        # 初始化智能体
        self.initialize_agent(model_name)
        
        # 开始对话
        self.chat_loop()


def main():
    """主函数"""
    try:
        cli = MathAgentCLI()
        cli.run()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
LangGraph 数学智能体演示脚本
"""

import os
import sys

# 设置环境变量
os.environ['DEEPSEEK_API_KEY'] = 'sk-797762b3a4ee44c8b315d748b36cc9f6'

# 添加 src 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import ModelManager
from src.chat_history import ChatHistoryManager
from src.agent import MathAgent


def demo():
    """演示数学智能体功能"""
    print("🎯 LangGraph 数学智能体演示")
    print("=" * 50)
    
    # 初始化组件
    model_manager = ModelManager()
    history_manager = ChatHistoryManager()
    model_manager.switch_model('deepseek')
    agent = MathAgent(model_manager, history_manager)
    
    print("✅ 智能体初始化完成！")
    print()
    
    # 演示问题
    questions = [
        "计算 25 + 37",
        "计算 144 除以 12", 
        "计算 7 乘以 8",
        "计算 100 减去 45"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"📝 问题 {i}: {question}")
        print("🤖 回答: ", end="", flush=True)
        
        # 流式输出
        for chunk in agent.chat_stream(question):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 50)
    
    print("\n🎉 演示完成！")
    print("💡 提示: 运行 'python main.py' 启动交互式 CLI")


if __name__ == "__main__":
    demo()


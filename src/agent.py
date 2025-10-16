"""
LangGraph Agent 核心逻辑
使用 StateGraph 构建智能体，支持工具调用和流式对话
"""

import time
from typing import Dict, Any, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 数学工具已移除，现在只使用数据分析工具
from .data_tools import (
    list_csv_files, load_csv_file, get_column_info, get_column_stats,
    calculate_summary, get_unique_values, filter_data, group_by_sum,
    group_by_aggregate, calculate_correlation, get_top_n_rows, search_rows,
    professional_data_analysis, generate_pdf_report, generate_html_to_pdf_report
)
from .models import ModelManager
from .chat_history import ChatHistoryManager
from .logger import get_logger, PerformanceTimer
from .model_call_logger import model_call_logger


class MathAgent:
    """数学智能体类"""
    
    def __init__(self, model_manager: ModelManager, history_manager: ChatHistoryManager):
        """
        初始化数学智能体
        
        Args:
            model_manager: 模型管理器
            history_manager: 历史管理器
        """
        self.logger = get_logger('agent')
        self.model_manager = model_manager
        self.history_manager = history_manager
        self.tools = [
            list_csv_files, load_csv_file, get_column_info, get_column_stats,
            calculate_summary, get_unique_values, filter_data, group_by_sum,
            group_by_aggregate, calculate_correlation, get_top_n_rows, search_rows,
            professional_data_analysis,  # 专业数据分析工具
            generate_pdf_report, generate_html_to_pdf_report  # PDF生成工具
        ]
        self.tool_node = self._create_tool_node()
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.logger.info("数学智能体初始化完成")
    
    def _create_tool_node(self):
        """创建带日志记录的工具节点"""
        from langgraph.prebuilt import ToolNode
        
        # 创建标准的ToolNode
        tool_node = ToolNode(self.tools)
        
        def logging_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """执行工具并记录日志"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # 记录工具执行开始
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    
                    # 生成调用ID（使用时间戳）
                    call_id = f"tool_{int(time.time() * 1000)}"
                    
                    model_call_logger.log_tool_execution_start(
                        call_id=call_id,
                        tool_name=tool_name,
                        tool_args=tool_args
                    )
                
                # 执行工具
                start_time = time.time()
                try:
                    # 调用标准的ToolNode
                    tool_result = tool_node.invoke(state)
                    execution_time = time.time() - start_time
                    
                    # 记录工具执行结果
                    model_call_logger.log_tool_execution_result(
                        call_id=call_id,
                        tool_name=tool_name,
                        result="工具执行成功",
                        execution_time=execution_time
                    )
                    
                    # 重要：保留原始消息序列，只添加ToolMessage
                    # ToolNode默认只返回ToolMessage，我们需要保留完整的消息历史
                    return {"messages": messages + tool_result["messages"]}
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    model_call_logger.log_tool_execution_error(
                        call_id=call_id,
                        tool_name=tool_name,
                        error=e,
                        execution_time=execution_time
                    )
                    raise
            else:
                # 没有工具调用，直接返回
                return state
        
        return logging_tool_node
    
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        
        # 定义状态类型
        from typing import TypedDict
        
        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], "消息列表"]
        
        # 创建状态图
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self.tool_node)
        
        # 设置入口点
        graph.set_entry_point("agent")
        
        # 添加边
        graph.add_edge("tools", "agent")
        
        # 添加条件边
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # 编译图
        return graph.compile(checkpointer=self.memory)
    
    def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        智能体节点 - 调用 LLM 生成响应
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        messages = state["messages"]
        llm = self.model_manager.get_current_llm()
        
        if not llm:
            raise ValueError("请先选择一个模型")
        
        # 将工具绑定到LLM
        llm_with_tools = llm.bind_tools(self.tools)
        
        # 构建系统提示
        system_prompt = """你是一个智能助手，可以帮助用户进行数据分析。

**数据分析工具：**
- list_csv_files() - 查看data文件夹中可用的CSV文件
- load_csv_file(filename) - 加载CSV文件并查看基本信息（行数、列数、列名、预览）
- get_column_info(filename, column_name) - 获取列的详细信息（类型、唯一值、缺失值）
- get_column_stats(filename, column_name) - 获取数值列的统计信息（均值、中位数、最大最小值、总和等）
- calculate_summary(filename) - 计算所有数值列的汇总统计
- get_unique_values(filename, column_name) - 获取列的所有唯一值
- filter_data(filename, column_name, operator, value) - 根据条件筛选数据
- group_by_sum(filename, group_column, sum_column) - 按列分组求和
- group_by_aggregate(filename, group_column, agg_column, agg_function) - 分组聚合分析（sum/mean/count/max/min）
- calculate_correlation(filename, column1, column2) - 计算两列的相关性
- get_top_n_rows(filename, column_name, n, ascending) - 获取排序后的前N行
- search_rows(filename, column_name, keyword) - 在列中搜索关键词
- professional_data_analysis(filename, analysis_type) - 专业数据分析工具，支持IPTV业务分析、趋势分析、同比环比分析等

**PDF报告生成工具：**
- generate_pdf_report(filename, output_dir, report_title) - 生成PDF格式的数据分析报告（使用ReportLab）
- generate_html_to_pdf_report(filename, output_dir, report_title) - 生成HTML转PDF格式的报告（使用WeasyPrint，支持更丰富的样式）

**数据分析流程建议：**
1. 先用list_csv_files查看可用文件
2. 用load_csv_file了解文件结构和列名
3. 根据用户问题选择合适的分析工具
4. 逐步分析并给出清晰的结论
5. 如需生成报告，使用PDF生成工具

当用户提出数据分析问题时，请：
1. 先了解数据结构
2. 选择合适的分析工具
3. 给出详细的分析结果和数据支持
4. 如果用户需要报告，主动建议生成PDF报告

当用户需要生成报告时，请：
1. 先进行数据分析
2. 使用generate_pdf_report或generate_html_to_pdf_report生成PDF报告
3. 告知用户报告已生成并提供文件路径

始终用中文回复。"""
        
        # 检查是否包含工具执行结果
        has_tool_result = any(isinstance(msg, ToolMessage) for msg in messages)
        
        if has_tool_result:
            # 如果包含工具执行结果，只发送最后几条消息（不包含SystemMessage）
            # 找到最后一个HumanMessage，从那里开始
            last_human_index = -1
            for i, msg in enumerate(messages):
                if isinstance(msg, HumanMessage):
                    last_human_index = i
            
            if last_human_index >= 0:
                messages = messages[last_human_index:]
        else:
            # 如果没有工具执行结果，添加系统消息
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages
            else:
                # 如果已经有SystemMessage，确保它包含正确的提示
                if isinstance(messages[0], SystemMessage):
                    messages[0] = SystemMessage(content=system_prompt)
        
        # 记录模型调用开始
        user_input = messages[-1].content if messages else "未知输入"
        session_id = getattr(state, 'session_id', 'unknown')
        model_name = self.model_manager.get_current_model()
        model_config = self.model_manager.get_model_info(model_name) if model_name else {}
        
        call_id = model_call_logger.log_model_call_start(
            model_name=model_name or "unknown",
            session_id=session_id,
            user_input=user_input,
            model_config=model_config
        )
        
        # 记录请求数据
        request_data = {
            "messages": [{"role": msg.__class__.__name__, "content": msg.content} for msg in messages],
            "model": model_name,
            "temperature": model_config.get('temperature', 0.7),
            "max_tokens": model_config.get('max_tokens', 2000)
        }
        model_call_logger.log_model_call_request(call_id, request_data)
        
        # 调用 LLM
        start_time = time.time()
        try:
            response = llm_with_tools.invoke(messages)
            response_time = time.time() - start_time
            
            # 记录响应数据
            response_data = {
                "content": response.content,
                "response_metadata": getattr(response, 'response_metadata', {}),
                "usage_metadata": getattr(response, 'usage_metadata', {})
            }
            model_call_logger.log_model_call_response(call_id, response_data, response_time)
            
            # 记录工具调用判断
            self._log_tool_decision(call_id, response)
            
        except Exception as e:
            error_time = time.time() - start_time
            model_call_logger.log_model_call_error(call_id, e, error_time)
            raise
        
        # 记录调用完成
        total_time = time.time() - start_time
        usage_metadata = getattr(response, 'usage_metadata', {})
        tokens_used = usage_metadata.get('total_tokens') if usage_metadata else None
        model_call_logger.log_model_call_complete(call_id, total_time, tokens_used)
        
        # 更新状态
        return {"messages": messages + [response]}
    
    def _log_tool_decision(self, call_id: str, response):
        """记录大模型对工具调用的判断"""
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # 有工具调用
            tool_calls_data = []
            for tool_call in response.tool_calls:
                tool_calls_data.append({
                    'name': tool_call.get('name', 'unknown'),
                    'args': tool_call.get('args', {})
                })
            
            model_call_logger.log_tool_decision(
                call_id=call_id,
                decision="use_tools",
                reasoning="模型决定使用工具来完成任务",
                tool_calls=tool_calls_data
            )
        else:
            # 直接回复
            model_call_logger.log_tool_decision(
                call_id=call_id,
                decision="direct_reply",
                reasoning="模型决定直接回复用户，无需使用工具"
            )
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """
        判断是否应该继续调用工具
        
        Args:
            state: 当前状态
            
        Returns:
            "continue" 或 "end"
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # 如果最后一条消息包含工具调用，则继续
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    def chat(self, user_input: str, session_id: str = None) -> str:
        """
        与智能体对话
        
        Args:
            user_input: 用户输入
            session_id: 会话ID，如果为None则创建新会话
            
        Returns:
            智能体响应
        """
        # 处理会话
        if session_id:
            if not self.history_manager.load_session(session_id):
                session_id = self.history_manager.create_session()
        else:
            session_id = self.history_manager.create_session()
        
        # 添加用户消息到历史
        self.history_manager.add_message("user", user_input)
        
        # 构建消息列表
        history = self.history_manager.get_current_history()
        messages = []
        
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        # 运行图
        initial_state = {"messages": messages}
        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config=config)
        
        # 获取响应
        response_messages = result["messages"]
        assistant_response = response_messages[-1].content
        
        # 保存助手响应到历史
        self.history_manager.add_message("assistant", assistant_response)
        self.history_manager.save_session()
        
        return assistant_response
    
    def chat_stream(self, user_input: str, session_id: str = None):
        """
        流式对话
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            
        Yields:
            流式响应块
        """
        # 处理会话
        if session_id:
            if not self.history_manager.load_session(session_id):
                session_id = self.history_manager.create_session()
        else:
            session_id = self.history_manager.create_session()
        
        # 添加用户消息到历史
        self.history_manager.add_message("user", user_input)
        
        # 构建消息列表
        history = self.history_manager.get_current_history()
        messages = []
        
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        # 流式运行图
        initial_state = {"messages": messages}
        config = {"configurable": {"thread_id": session_id}}
        
        full_response = ""
        for chunk in self.graph.stream(initial_state, config=config):
            if "agent" in chunk:
                agent_messages = chunk["agent"]["messages"]
                if agent_messages:
                    last_message = agent_messages[-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        content = last_message.content
                        # 检查是否是新的内容
                        if content != full_response:
                            # 计算新增内容
                            if full_response and content.startswith(full_response):
                                new_content = content[len(full_response):]
                            else:
                                new_content = content
                            full_response = content
                            # 只输出新增内容
                            if new_content:
                                yield new_content
        
        # 保存完整响应到历史
        self.history_manager.add_message("assistant", full_response)
        self.history_manager.save_session()
    
    def get_available_tools(self) -> List[BaseTool]:
        """获取可用工具列表"""
        return self.tools
    
    def get_current_session_id(self) -> str:
        """获取当前会话ID"""
        return self.history_manager.current_session_id

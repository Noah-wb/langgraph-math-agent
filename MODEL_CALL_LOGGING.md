# 模型调用日志功能说明

## 功能概述

本项目已成功实现了完整的模型调用日志功能，包括大模型判断是否需要调用工具的详细日志记录。

## 主要功能

### 1. 模型调用日志记录器 (`src/model_call_logger.py`)

专门用于记录智能体与大模型之间的交互详情，包括：

- **模型调用开始/结束**：记录每次模型调用的开始时间、模型名称、会话ID等
- **请求/响应数据**：记录发送给模型的请求数据和收到的响应数据
- **工具调用判断**：记录大模型是否决定使用工具以及具体使用哪些工具
- **工具执行过程**：记录工具的执行开始、结果和错误
- **模型切换**：记录模型切换的详细信息
- **性能指标**：记录响应时间、token消耗等

### 2. 日志记录方法

#### 模型调用相关
- `log_model_call_start()` - 记录模型调用开始
- `log_model_call_request()` - 记录请求数据
- `log_model_call_response()` - 记录响应数据
- `log_model_call_error()` - 记录调用错误
- `log_model_call_complete()` - 记录调用完成

#### 工具调用判断相关
- `log_tool_decision()` - 记录大模型对工具调用的判断
- `log_tool_execution_start()` - 记录工具执行开始
- `log_tool_execution_result()` - 记录工具执行结果
- `log_tool_execution_error()` - 记录工具执行错误

#### 其他功能
- `log_model_switch()` - 记录模型切换
- `log_model_health_check()` - 记录模型健康检查

### 3. 智能体集成 (`src/agent.py`)

智能体已完全集成模型调用日志功能：

- **自动记录**：每次模型调用都会自动记录详细信息
- **工具判断日志**：自动记录模型是否决定使用工具
- **自定义工具节点**：创建了带日志记录的工具执行节点
- **性能监控**：记录每次调用的响应时间和token消耗

### 4. 模型管理器集成 (`src/models.py`)

模型管理器已集成日志功能：

- **模型切换日志**：记录模型切换的详细信息
- **配置信息记录**：记录模型配置参数

## 日志文件

### 主要日志文件
- `logs/model_calls.log` - 模型调用主要日志
- `logs/model_details.log` - 详细调试日志

### 日志格式示例

```
2025-10-15 16:01:27 [INFO] model_calls: 🚀 模型调用开始 - ID: call_1760515284787, 模型: deepseek, 会话: unknown
2025-10-15 16:01:27 [INFO] model_calls: 📤 发送请求 - 调用ID: call_1760515284787
2025-10-15 16:01:27 [INFO] model_calls: 📥 收到响应 - 调用ID: call_1760515284787, 耗时: 2.949s
2025-10-15 16:01:27 [INFO] model_calls: 🔧 决定使用工具 - 调用ID: call_1760515284787
2025-10-15 16:01:27 [INFO] model_calls:   工具 1: multiply({'a': 15, 'b': 8})
2025-10-15 16:01:27 [INFO] model_calls: ✅ 调用完成 - 调用ID: call_1760515284787, 总耗时: 2.950s, 消耗tokens: 668
```

## 工具调用判断日志

### 判断类型
1. **使用工具** (`🔧 决定使用工具`)
   - 记录模型决定使用工具
   - 显示具体要调用的工具名称和参数
   - 记录工具执行过程

2. **直接回复** (`💬 直接回复用户`)
   - 记录模型决定直接回复用户
   - 无需使用工具的情况

### 日志示例

#### 使用工具的情况
```
🔧 决定使用工具 - 调用ID: call_xxx
  工具 1: multiply({'a': 15, 'b': 8})
⚙️ 开始执行工具 - 调用ID: tool_xxx, 工具: multiply
✅ 工具执行完成 - 调用ID: tool_xxx, 工具: multiply, 耗时: 0.001s
```

#### 直接回复的情况
```
💬 直接回复用户 - 调用ID: call_xxx
```

## 测试验证

### 测试脚本
- `test_model_calls.py` - 基础模型调用日志测试
- `test_tool_decision.py` - 工具调用判断日志测试

### 测试场景
1. **数学计算问题** - 测试工具调用判断
2. **非数学问题** - 测试直接回复判断
3. **复杂计算** - 测试多步骤工具调用
4. **错误处理** - 测试工具执行错误日志

## 使用方式

### 自动记录
模型调用日志功能已完全自动化，无需手动配置：

```python
# 智能体会自动记录所有模型调用和工具判断
agent = MathAgent(model_manager, history_manager)
result = agent.chat("计算 5 + 3", session_id)
```

### 查看日志
```python
from src.model_call_logger import model_call_logger

# 获取调用历史
history = model_call_logger.get_call_history(session_id)

# 查看日志文件
# logs/model_calls.log - 主要日志
# logs/model_details.log - 详细日志
```

## 技术特点

1. **完整性**：覆盖模型调用的完整生命周期
2. **详细性**：记录请求、响应、工具判断、执行结果等详细信息
3. **性能监控**：记录响应时间、token消耗等性能指标
4. **错误处理**：完整的错误日志记录
5. **可扩展性**：易于添加新的日志记录功能

## 问题修复

### 工具执行消息格式问题

在实现过程中遇到了一个关键问题：工具执行后的消息格式错误导致模型调用失败。

**问题描述：**
```
Error code: 400 - {'error': {'message': "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'", 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
```

**问题原因：**
1. 工具执行节点返回的ToolMessage格式不正确
2. 消息序列中包含了不必要的SystemMessage
3. 工具调用ID匹配问题

**解决方案：**
1. **修复工具执行节点**：使用标准的ToolNode并正确调用其invoke方法
2. **保留完整消息序列**：修复ToolNode默认行为，确保保留原始消息历史
3. **正确消息配对**：确保ToolMessage与对应的AIMessage正确配对
4. **工具调用ID匹配**：确保ToolMessage的tool_call_id与AIMessage的tool_calls匹配

**修复后的效果：**
- ✅ 工具执行成功记录日志
- ✅ 消息序列正确处理（HumanMessage -> AIMessage -> ToolMessage）
- ✅ 模型能正确接收工具执行结果
- ✅ 完整的工具调用流程正常工作
- ✅ 工具调用ID正确匹配
- ✅ 保留完整的对话历史上下文
- ✅ 流式输出正确显示计算结果
- ✅ 日志输出不干扰用户界面

## 总结

模型调用日志功能已完全实现，包括：
- ✅ 模型调用生命周期日志
- ✅ 工具调用判断日志
- ✅ 工具执行过程日志
- ✅ 性能监控日志
- ✅ 错误处理日志
- ✅ 模型切换日志
- ✅ 工具执行消息格式问题修复
- ✅ 流式输出正确显示计算结果
- ✅ 用户界面优化（日志不干扰显示）

该功能为智能体的调试、监控和优化提供了完整的日志支持。

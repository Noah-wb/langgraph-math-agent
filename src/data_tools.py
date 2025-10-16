"""
CSV数据分析工具模块
提供基础和高級数据分析功能，支持中文列名和复杂数据结构
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from langchain.tools import tool
import warnings
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from datetime import datetime
import markdown
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from weasyprint import HTML, CSS

# 忽略pandas警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# 数据文件夹路径
DATA_DIR = "./data"

def _validate_file_path(filename: str) -> str:
    """验证文件路径，确保文件在data文件夹内"""
    # 移除路径分隔符，防止路径遍历攻击
    safe_filename = os.path.basename(filename)
    full_path = os.path.join(DATA_DIR, safe_filename)
    
    # 确保路径在data文件夹内
    if not os.path.abspath(full_path).startswith(os.path.abspath(DATA_DIR)):
        raise ValueError("文件路径不安全，只能访问data文件夹内的文件")
    
    return full_path

def _safe_read_csv(filename: str) -> pd.DataFrame:
    """安全读取CSV文件"""
    try:
        file_path = _validate_file_path(filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        # 读取CSV文件，支持中文编码
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except Exception as e:
        raise Exception(f"读取CSV文件失败: {str(e)}")

@tool
def list_csv_files() -> str:
    """
    列出data文件夹中的所有CSV文件
    
    Returns:
        str: 可用CSV文件列表的格式化字符串
    """
    try:
        if not os.path.exists(DATA_DIR):
            return "data文件夹不存在"
        
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        if not csv_files:
            return "data文件夹中没有CSV文件"
        
        result = "📁 可用的CSV文件:\n"
        for i, file in enumerate(csv_files, 1):
            file_path = os.path.join(DATA_DIR, file)
            file_size = os.path.getsize(file_path)
            result += f"  {i}. {file} ({file_size:,} bytes)\n"
        
        return result
    except Exception as e:
        return f"❌ 列出CSV文件失败: {str(e)}"

@tool
def load_csv_file(filename: str) -> str:
    """
    加载CSV文件并返回基本信息
    
    Args:
        filename: CSV文件名
        
    Returns:
        str: 文件基本信息的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        result = f"📊 文件: {filename}\n"
        result += f"📏 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列\n"
        result += f"📋 列名: {', '.join(df.columns.tolist())}\n\n"
        
        # 显示前5行数据
        result += "📄 前5行数据预览:\n"
        preview = df.head().to_string(index=False)
        result += preview
        
        return result
    except Exception as e:
        return f"❌ 加载CSV文件失败: {str(e)}"

@tool
def get_column_info(filename: str, column_name: str) -> str:
    """
    获取指定列的详细信息
    
    Args:
        filename: CSV文件名
        column_name: 列名
        
    Returns:
        str: 列详细信息的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        col = df[column_name]
        
        result = f"📊 列名: {column_name}\n"
        result += f"📏 数据类型: {col.dtype}\n"
        result += f"📈 非空值数量: {col.count()}\n"
        result += f"❌ 缺失值数量: {col.isnull().sum()}\n"
        result += f"🔢 唯一值数量: {col.nunique()}\n"
        
        if col.dtype in ['int64', 'float64']:
            result += f"📊 数值范围: {col.min()} ~ {col.max()}\n"
        
        # 显示前10个唯一值
        unique_values = col.dropna().unique()[:10]
        result += f"🔍 前10个唯一值: {', '.join(map(str, unique_values))}\n"
        
        return result
    except Exception as e:
        return f"❌ 获取列信息失败: {str(e)}"

@tool
def get_column_stats(filename: str, column_name: str) -> str:
    """
    获取数值列的统计信息
    
    Args:
        filename: CSV文件名
        column_name: 列名
        
    Returns:
        str: 统计信息的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        col = df[column_name]
        
        if not pd.api.types.is_numeric_dtype(col):
            return f"❌ 列 '{column_name}' 不是数值类型，无法计算统计信息"
        
        # 移除缺失值
        col_clean = col.dropna()
        
        if len(col_clean) == 0:
            return f"❌ 列 '{column_name}' 没有有效的数值数据"
        
        result = f"📊 列 '{column_name}' 统计信息:\n"
        result += f"📏 数据点数量: {len(col_clean)}\n"
        result += f"➕ 总和: {col_clean.sum():,.2f}\n"
        result += f"📊 平均值: {col_clean.mean():,.2f}\n"
        result += f"📈 中位数: {col_clean.median():,.2f}\n"
        result += f"📉 最小值: {col_clean.min():,.2f}\n"
        result += f"📈 最大值: {col_clean.max():,.2f}\n"
        result += f"📊 标准差: {col_clean.std():,.2f}\n"
        result += f"📊 方差: {col_clean.var():,.2f}\n"
        
        return result
    except Exception as e:
        return f"❌ 获取统计信息失败: {str(e)}"

@tool
def calculate_summary(filename: str) -> str:
    """
    计算所有数值列的汇总统计
    
    Args:
        filename: CSV文件名
        
    Returns:
        str: 汇总统计的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "❌ 文件中没有数值列"
        
        result = f"📊 数值列汇总统计 (共{len(numeric_cols)}列):\n\n"
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                result += f"📈 {col}:\n"
                result += f"  总和: {col_data.sum():,.2f}\n"
                result += f"  平均值: {col_data.mean():,.2f}\n"
                result += f"  中位数: {col_data.median():,.2f}\n"
                result += f"  标准差: {col_data.std():,.2f}\n\n"
        
        return result
    except Exception as e:
        return f"❌ 计算汇总统计失败: {str(e)}"

@tool
def get_unique_values(filename: str, column_name: str) -> str:
    """
    获取列的所有唯一值
    
    Args:
        filename: CSV文件名
        column_name: 列名
        
    Returns:
        str: 唯一值列表的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        unique_vals = df[column_name].dropna().unique()
        
        result = f"🔍 列 '{column_name}' 的唯一值 (共{len(unique_vals)}个):\n"
        
        # 如果唯一值太多，只显示前20个
        if len(unique_vals) > 20:
            result += f"显示前20个唯一值:\n"
            for val in unique_vals[:20]:
                result += f"  • {val}\n"
            result += f"... 还有 {len(unique_vals) - 20} 个值"
        else:
            for val in unique_vals:
                result += f"  • {val}\n"
        
        return result
    except Exception as e:
        return f"❌ 获取唯一值失败: {str(e)}"

@tool
def filter_data(filename: str, column_name: str, operator: str, value: Union[str, int, float]) -> str:
    """
    根据条件筛选数据
    
    Args:
        filename: CSV文件名
        column_name: 列名
        operator: 操作符 (>, <, =, >=, <=, !=, contains)
        value: 比较值
        
    Returns:
        str: 筛选结果的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        # 根据操作符筛选数据
        if operator == ">":
            filtered_df = df[df[column_name] > value]
        elif operator == "<":
            filtered_df = df[df[column_name] < value]
        elif operator == "=":
            filtered_df = df[df[column_name] == value]
        elif operator == ">=":
            filtered_df = df[df[column_name] >= value]
        elif operator == "<=":
            filtered_df = df[df[column_name] <= value]
        elif operator == "!=":
            filtered_df = df[df[column_name] != value]
        elif operator == "contains":
            filtered_df = df[df[column_name].astype(str).str.contains(str(value), na=False)]
        else:
            return f"❌ 不支持的操作符: {operator}。支持的操作符: >, <, =, >=, <=, !=, contains"
        
        result = f"🔍 筛选条件: {column_name} {operator} {value}\n"
        result += f"📊 筛选结果: {len(filtered_df)} 行 (原数据 {len(df)} 行)\n\n"
        
        if len(filtered_df) > 0:
            result += "📄 筛选结果预览 (前10行):\n"
            preview = filtered_df.head(10).to_string(index=False)
            result += preview
        else:
            result += "❌ 没有找到符合条件的数据"
        
        return result
    except Exception as e:
        return f"❌ 筛选数据失败: {str(e)}"

@tool
def group_by_sum(filename: str, group_column: str, sum_column: str) -> str:
    """
    按列分组并求和
    
    Args:
        filename: CSV文件名
        group_column: 分组列名
        sum_column: 求和列名
        
    Returns:
        str: 分组求和结果的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if group_column not in df.columns:
            return f"❌ 分组列 '{group_column}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if sum_column not in df.columns:
            return f"❌ 求和列 '{sum_column}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if not pd.api.types.is_numeric_dtype(df[sum_column]):
            return f"❌ 求和列 '{sum_column}' 不是数值类型"
        
        # 分组求和
        grouped = df.groupby(group_column)[sum_column].sum().reset_index()
        grouped = grouped.sort_values(sum_column, ascending=False)
        
        result = f"📊 按 '{group_column}' 分组，对 '{sum_column}' 求和:\n\n"
        result += grouped.to_string(index=False)
        
        result += f"\n\n📈 总计: {grouped[sum_column].sum():,.2f}"
        
        return result
    except Exception as e:
        return f"❌ 分组求和失败: {str(e)}"

@tool
def group_by_aggregate(filename: str, group_column: str, agg_column: str, agg_function: str) -> str:
    """
    分组聚合分析
    
    Args:
        filename: CSV文件名
        group_column: 分组列名
        agg_column: 聚合列名
        agg_function: 聚合函数 (sum, mean, count, max, min)
        
    Returns:
        str: 聚合结果的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if group_column not in df.columns:
            return f"❌ 分组列 '{group_column}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if agg_column not in df.columns:
            return f"❌ 聚合列 '{agg_column}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if agg_function not in ['sum', 'mean', 'count', 'max', 'min']:
            return f"❌ 不支持的聚合函数: {agg_function}。支持的函数: sum, mean, count, max, min"
        
        # 执行聚合
        if agg_function == 'sum':
            result_series = df.groupby(group_column)[agg_column].sum()
        elif agg_function == 'mean':
            result_series = df.groupby(group_column)[agg_column].mean()
        elif agg_function == 'count':
            result_series = df.groupby(group_column)[agg_column].count()
        elif agg_function == 'max':
            result_series = df.groupby(group_column)[agg_column].max()
        elif agg_function == 'min':
            result_series = df.groupby(group_column)[agg_column].min()
        
        result_df = result_series.reset_index()
        result_df = result_df.sort_values(agg_column, ascending=False)
        
        result = f"📊 按 '{group_column}' 分组，对 '{agg_column}' 执行 {agg_function} 聚合:\n\n"
        result += result_df.to_string(index=False)
        
        return result
    except Exception as e:
        return f"❌ 分组聚合失败: {str(e)}"

@tool
def calculate_correlation(filename: str, column1: str, column2: str) -> str:
    """
    计算两个数值列的相关系数
    
    Args:
        filename: CSV文件名
        column1: 第一个列名
        column2: 第二个列名
        
    Returns:
        str: 相关系数的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column1 not in df.columns:
            return f"❌ 列 '{column1}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if column2 not in df.columns:
            return f"❌ 列 '{column2}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if not pd.api.types.is_numeric_dtype(df[column1]):
            return f"❌ 列 '{column1}' 不是数值类型"
        
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return f"❌ 列 '{column2}' 不是数值类型"
        
        # 计算相关系数
        correlation = df[column1].corr(df[column2])
        
        result = f"📊 列 '{column1}' 和 '{column2}' 的相关系数:\n"
        result += f"🔗 相关系数: {correlation:.4f}\n\n"
        
        # 解释相关系数
        if abs(correlation) >= 0.8:
            strength = "强"
        elif abs(correlation) >= 0.5:
            strength = "中等"
        elif abs(correlation) >= 0.3:
            strength = "弱"
        else:
            strength = "很弱"
        
        direction = "正" if correlation > 0 else "负"
        result += f"📈 相关性: {strength}{direction}相关"
        
        return result
    except Exception as e:
        return f"❌ 计算相关系数失败: {str(e)}"

@tool
def get_top_n_rows(filename: str, column_name: str, n: int, ascending: bool = False) -> str:
    """
    按指定列排序获取前N行
    
    Args:
        filename: CSV文件名
        column_name: 排序列名
        n: 返回行数
        ascending: 是否升序排列 (默认False，即降序)
        
    Returns:
        str: 排序结果的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return f"❌ 列 '{column_name}' 不是数值类型，无法排序"
        
        # 排序并获取前N行
        sorted_df = df.sort_values(column_name, ascending=ascending).head(n)
        
        order = "升序" if ascending else "降序"
        result = f"📊 按 '{column_name}' {order}排列的前{n}行:\n\n"
        result += sorted_df.to_string(index=False)
        
        return result
    except Exception as e:
        return f"❌ 获取排序结果失败: {str(e)}"

@tool
def search_rows(filename: str, column_name: str, keyword: str) -> str:
    """
    在列中搜索包含关键词的行
    
    Args:
        filename: CSV文件名
        column_name: 搜索列名
        keyword: 搜索关键词
        
    Returns:
        str: 搜索结果的格式化字符串
    """
    try:
        df = _safe_read_csv(filename)
        
        if column_name not in df.columns:
            return f"❌ 列 '{column_name}' 不存在。可用列: {', '.join(df.columns.tolist())}"
        
        # 搜索包含关键词的行
        mask = df[column_name].astype(str).str.contains(keyword, case=False, na=False)
        search_results = df[mask]
        
        result = f"🔍 在列 '{column_name}' 中搜索 '{keyword}':\n"
        result += f"📊 找到 {len(search_results)} 行匹配结果\n\n"
        
        if len(search_results) > 0:
            result += "📄 搜索结果:\n"
            result += search_results.to_string(index=False)
        else:
            result += "❌ 没有找到包含该关键词的行"
        
        return result
    except Exception as e:
        return f"❌ 搜索失败: {str(e)}"

@tool
def professional_data_analysis(filename: str, analysis_type: str = "comprehensive") -> str:
    """
    专业数据分析工具，专门用于IPTV业务数据分析
    
    Args:
        filename: CSV文件名
        analysis_type: 分析类型 ("comprehensive", "daily_trend", "comparison", "summary")
    
    Returns:
        str: 专业数据分析报告
    """
    try:
        df = _safe_read_csv(filename)
        
        if df.empty:
            return "❌ 数据文件为空，无法进行分析"
        
        # 检查数据格式并适配
        result = "📊 专业数据分析报告\n"
        result += "=" * 50 + "\n\n"
        
        # 检查是否为IPTV产品包数据格式
        if '产品包名称' in df.columns and '产品包分类' in df.columns:
            # IPTV产品包数据格式
            return _analyze_iptv_product_data(df, analysis_type)
        else:
            # 检查是否为时间序列数据格式
            required_columns = ['日期', '产品包名称', '订购量']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return f"❌ 数据格式不支持，缺少必要的列: {', '.join(missing_columns)}"
            return _analyze_time_series_data(df, analysis_type)
        
        
    except Exception as e:
        return f"❌ 专业数据分析失败: {str(e)}"


def _analyze_iptv_product_data(df: pd.DataFrame, analysis_type: str) -> str:
    """分析IPTV产品包数据 - 使用AI模型和提示词进行专业分析"""
    try:
        # 读取提示词文件
        prompt_file = "prompts/data_analysis_prompt.txt"
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        else:
            # 如果提示词文件不存在，使用默认提示词
            prompt_template = """你是一个专业的数据分析大师，专注于IPTV业务领域。请对提供的数据进行专业分析，包括：
1. 数据走势分析
2. 同比环比分析  
3. 总结性分析
请提供详细、专业的分析报告。"""
        
        # 准备数据摘要
        data_summary = _prepare_data_summary(df)
        
        # 构建完整的提示词
        full_prompt = f"{prompt_template}\n\n数据摘要：\n{data_summary}\n\n请按照提示词要求进行专业的数据分析。"
        
        # 使用AI模型进行分析
        from src.models import ModelManager
        model_manager = ModelManager()
        
        try:
            # 切换到默认模型（如果还没有设置）
            if not model_manager.get_current_llm():
                # 尝试切换到第一个可用模型
                available_models = model_manager.get_available_models()
                if available_models:
                    model_manager.switch_model(available_models[0])
                else:
                    raise Exception("没有可用的模型")
            
            # 获取当前LLM实例
            llm = model_manager.get_current_llm()
            if not llm:
                raise Exception("无法获取LLM实例")
            
            # 构建消息格式
            messages = [{"role": "user", "content": full_prompt}]
            
            # 调用模型进行分析
            response = llm.invoke(messages)
            
            if response and hasattr(response, 'content') and response.content.strip():
                return f"📊 专业数据分析报告\n{'='*50}\n\n{response.content}\n\n{'='*50}"
            else:
                # 如果AI分析失败，回退到基础分析
                return _fallback_analysis(df)
                
        except Exception as e:
            logging.warning(f"AI模型分析失败，使用基础分析: {str(e)}")
            return _fallback_analysis(df)
            
    except Exception as e:
        logging.error(f"专业数据分析失败: {str(e)}")
        return _fallback_analysis(df)


def _prepare_data_summary(df: pd.DataFrame) -> str:
    """准备数据摘要供AI分析使用"""
    summary = []
    
    # 基础统计
    total_products = len(df)
    summary.append(f"产品包总数: {total_products}")
    
    # 分类统计
    if '产品包分类' in df.columns:
        category_stats = df.groupby('产品包分类').agg({
            '产品包名称': 'count',
            '产品包单价（元）': ['mean', 'sum'],
            '2025年度6月 新增订购（单）': 'sum',
            '2025年度6月 流水（元）\n（单价×新增订购）': 'sum'
        }).round(2)
        
        summary.append("\n分类统计:")
        for category in category_stats.index:
            count = category_stats.loc[category, ('产品包名称', 'count')]
            avg_price = category_stats.loc[category, ('产品包单价（元）', 'mean')]
            total_orders = category_stats.loc[category, ('2025年度6月 新增订购（单）', 'sum')]
            total_revenue = category_stats.loc[category, ('2025年度6月 流水（元）\n（单价×新增订购）', 'sum')]
            
            summary.append(f"  {category}: 产品数{count}, 平均单价¥{avg_price:.2f}, 订购量{total_orders:,}, 流水¥{total_revenue:,.2f}")
    
    # 热门产品
    if '2025年度6月 新增订购（单）' in df.columns:
        top_products = df.nlargest(5, '2025年度6月 新增订购（单）')
        summary.append("\n热门产品TOP5:")
        for i, (_, row) in enumerate(top_products.iterrows(), 1):
            summary.append(f"  {i}. {row['产品包名称']}: 订购量{row['2025年度6月 新增订购（单）']:,}")
    
    # 增长率分析
    if '订购同比增长率' in df.columns:
        growth_data = df[df['订购同比增长率'] != '-100.00%'].copy()
        if not growth_data.empty:
            summary.append("\n同比增长分析:")
            positive_count = 0
            negative_count = 0
            for _, row in growth_data.iterrows():
                try:
                    rate = float(row['订购同比增长率'].replace('%', ''))
                    if rate > 0:
                        positive_count += 1
                    elif rate < 0:
                        negative_count += 1
                except:
                    pass
            summary.append(f"  正增长: {positive_count}个, 负增长: {negative_count}个")
    
    return "\n".join(summary)


def _fallback_analysis(df: pd.DataFrame) -> str:
    """回退的基础分析（原分析逻辑）"""
    result = "📊 IPTV产品包数据分析报告\n"
    result += "=" * 50 + "\n\n"
    
    # 基础统计信息
    total_products = len(df)
    result += f"📈 产品包总数: {total_products}\n"
    
    # 按分类分析
    if '产品包分类' in df.columns:
        category_analysis = df.groupby('产品包分类').agg({
            '产品包名称': 'count',
            '产品包单价（元）': ['mean', 'sum'],
            '2025年度6月 新增订购（单）': 'sum',
            '2025年度6月 流水（元）\n（单价×新增订购）': 'sum'
        }).round(2)
        
        result += "\n1. 产品包分类分析\n"
        result += "-" * 30 + "\n"
        for category in category_analysis.index:
            count = category_analysis.loc[category, ('产品包名称', 'count')]
            avg_price = category_analysis.loc[category, ('产品包单价（元）', 'mean')]
            total_orders = category_analysis.loc[category, ('2025年度6月 新增订购（单）', 'sum')]
            total_revenue = category_analysis.loc[category, ('2025年度6月 流水（元）\n（单价×新增订购）', 'sum')]
            
            result += f"📦 {category}:\n"
            result += f"   • 产品数量: {count}\n"
            result += f"   • 平均单价: ¥{avg_price:.2f}\n"
            result += f"   • 总订购量: {total_orders:,}\n"
            result += f"   • 总流水: ¥{total_revenue:,.2f}\n\n"
    
    # 热门产品分析
    if '2025年度6月 新增订购（单）' in df.columns:
        top_products = df.nlargest(5, '2025年度6月 新增订购（单）')
        result += "2. 热门产品TOP5\n"
        result += "-" * 30 + "\n"
        for i, (_, row) in enumerate(top_products.iterrows(), 1):
            product_name = row['产品包名称']
            orders = row['2025年度6月 新增订购（单）']
            revenue = row['2025年度6月 流水（元）\n（单价×新增订购）']
            result += f"{i}. {product_name}\n"
            result += f"   • 订购量: {orders:,}\n"
            result += f"   • 流水: ¥{revenue:,.2f}\n\n"
    
    # 同比增长分析
    if '订购同比增长率' in df.columns:
        growth_analysis = df[df['订购同比增长率'] != '-100.00%'].copy()
        if not growth_analysis.empty:
            # 解析增长率数据
            growth_rates = []
            for rate_str in growth_analysis['订购同比增长率']:
                try:
                    rate = float(rate_str.replace('%', ''))
                    growth_rates.append(rate)
                except:
                    growth_rates.append(0)
            
            growth_analysis['growth_rate_num'] = growth_rates
            positive_growth = growth_analysis[growth_analysis['growth_rate_num'] > 0]
            negative_growth = growth_analysis[growth_analysis['growth_rate_num'] < 0]
            
            result += "3. 同比增长分析\n"
            result += "-" * 30 + "\n"
            result += f"📈 正增长产品: {len(positive_growth)}个\n"
            result += f"📉 负增长产品: {len(negative_growth)}个\n\n"
            
            if not positive_growth.empty:
                top_growth = positive_growth.nlargest(3, 'growth_rate_num')
                result += "增长最快产品TOP3:\n"
                for i, (_, row) in enumerate(top_growth.iterrows(), 1):
                    result += f"{i}. {row['产品包名称']}: +{row['growth_rate_num']:.1f}%\n"
                result += "\n"
    
    result += "=" * 50
    return result


def _analyze_time_series_data(df: pd.DataFrame, analysis_type: str) -> str:
    """分析时间序列数据"""
    result = "📊 时间序列数据分析报告\n"
    result += "=" * 50 + "\n\n"
    
    # 数据预处理
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    
    # 获取最新一天的数据
    latest_date = df['日期'].max()
    latest_data = df[df['日期'] == latest_date]
    
    # 计算当月数据
    current_month = latest_date.month
    current_year = latest_date.year
    monthly_data = df[(df['日期'].dt.month == current_month) & 
                     (df['日期'].dt.year == current_year)]
    
    # 计算每日平均值
    days_in_month = monthly_data['日期'].nunique()
    monthly_total = monthly_data['订购量'].sum()
    daily_average = monthly_total / days_in_month if days_in_month > 0 else 0
    
    result += f"📅 分析日期: {latest_date.strftime('%Y-%m-%d')}\n"
    result += f"📈 当月订购总量: {monthly_total:,}\n"
    result += f"📊 当月每日平均订购量: {daily_average:.2f}\n"
    result += f"📅 当月分析天数: {days_in_month}天\n\n"
    
    # 各类型产品包分析
    if '产品包类型' in df.columns:
        type_analysis = latest_data.groupby('产品包类型')['订购量'].sum().sort_values(ascending=False)
        result += "1. 各类型产品包订购量分析:\n"
        for ptype, amount in type_analysis.items():
            result += f"   • {ptype}: {amount:,} 订购量\n"
        result += "\n"
    
    result += "=" * 50
    return result


def generate_charts_for_analysis(filename: str, output_dir: str = "sjfx") -> List[str]:
    """
    为数据分析生成图表
    
    Args:
        filename: CSV文件名
        output_dir: 输出目录
    
    Returns:
        List[str]: 生成的图表文件路径列表
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        df = _safe_read_csv(filename)
        if df.empty:
            return []
        
        chart_files = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 检查是否为IPTV产品包数据格式
        if '产品包名称' in df.columns and '产品包分类' in df.columns:
            chart_files.extend(_generate_iptv_charts(df, output_dir))
        else:
            chart_files.extend(_generate_time_series_charts(df, output_dir))
        
        return chart_files
        
    except Exception as e:
        logging.error(f"生成图表失败: {str(e)}")
        return []


def _generate_iptv_charts(df: pd.DataFrame, output_dir: str) -> List[str]:
    """生成IPTV产品包数据图表"""
    chart_files = []
    
    try:
        # 1. 产品包分类订购量饼图
        if '产品包分类' in df.columns and '2025年度6月 新增订购（单）' in df.columns:
            category_orders = df.groupby('产品包分类')['2025年度6月 新增订购（单）'].sum()
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_orders)))
            wedges, texts, autotexts = plt.pie(category_orders.values, 
                                              labels=category_orders.index,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.title('产品包分类订购量分布', fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            
            chart_file = os.path.join(output_dir, 'category_orders_pie.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart_file)
        
        # 2. 热门产品TOP10柱状图
        if '2025年度6月 新增订购（单）' in df.columns:
            top_products = df.nlargest(10, '2025年度6月 新增订购（单）')
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(top_products)), top_products['2025年度6月 新增订购（单）'])
            plt.xlabel('产品包', fontsize=12)
            plt.ylabel('订购量（单）', fontsize=12)
            plt.title('热门产品TOP10订购量', fontsize=16, fontweight='bold')
            
            # 设置x轴标签
            plt.xticks(range(len(top_products)), 
                      [name[:10] + '...' if len(name) > 10 else name 
                       for name in top_products['产品包名称']], 
                      rotation=45, ha='right')
            
            # 在柱子上添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            chart_file = os.path.join(output_dir, 'top_products_bar.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart_file)
        
        # 3. 价格分布直方图
        if '产品包单价（元）' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['产品包单价（元）'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('产品包单价（元）', fontsize=12)
            plt.ylabel('产品数量', fontsize=12)
            plt.title('产品包价格分布', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            chart_file = os.path.join(output_dir, 'price_distribution.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart_file)
        
        # 4. 同比增长率分析
        if '订购同比增长率' in df.columns:
            # 解析增长率数据
            growth_rates = []
            valid_products = []
            for _, row in df.iterrows():
                try:
                    rate_str = str(row['订购同比增长率'])
                    if rate_str != '-100.00%' and rate_str != 'nan':
                        rate = float(rate_str.replace('%', ''))
                        growth_rates.append(rate)
                        valid_products.append(row['产品包名称'])
                except:
                    continue
            
            if growth_rates:
                plt.figure(figsize=(12, 8))
                colors = ['green' if x > 0 else 'red' for x in growth_rates]
                bars = plt.bar(range(len(growth_rates)), growth_rates, color=colors, alpha=0.7)
                
                plt.xlabel('产品包', fontsize=12)
                plt.ylabel('同比增长率 (%)', fontsize=12)
                plt.title('产品包同比增长率分析', fontsize=16, fontweight='bold')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 设置x轴标签
                plt.xticks(range(len(valid_products)), 
                          [name[:8] + '...' if len(name) > 8 else name 
                           for name in valid_products], 
                          rotation=45, ha='right')
                
                # 添加数值标签
                for i, (bar, rate) in enumerate(zip(bars, growth_rates)):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., 
                            height + (1 if height >= 0 else -3),
                            f'{rate:.1f}%', ha='center', 
                            va='bottom' if height >= 0 else 'top', fontsize=9)
                
                plt.tight_layout()
                chart_file = os.path.join(output_dir, 'growth_rate_analysis.png')
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart_file)
        
    except Exception as e:
        logging.error(f"生成IPTV图表失败: {str(e)}")
    
    return chart_files


def _generate_time_series_charts(df: pd.DataFrame, output_dir: str) -> List[str]:
    """生成时间序列数据图表"""
    chart_files = []
    
    try:
        # 时间序列趋势图
        if '日期' in df.columns and '订购量' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            daily_orders = df.groupby('日期')['订购量'].sum().sort_index()
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_orders.index, daily_orders.values, marker='o', linewidth=2, markersize=6)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('订购量', fontsize=12)
            plt.title('每日订购量趋势', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            chart_file = os.path.join(output_dir, 'daily_trend.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart_file)
    
    except Exception as e:
        logging.error(f"生成时间序列图表失败: {str(e)}")
    
    return chart_files


def _generate_text_analysis(filename: str) -> str:
    """生成文本分析内容（非LangChain工具版本）"""
    try:
        df = _safe_read_csv(filename)
        if df.empty:
            return "数据文件为空或无法读取。"
        
        # 检查是否为IPTV产品包数据格式
        if '产品包名称' in df.columns and '产品包分类' in df.columns:
            return _analyze_iptv_product_data(df, "comprehensive")
        else:
            return _analyze_time_series_data(df, "comprehensive")
            
    except Exception as e:
        logging.error(f"生成文本分析失败: {str(e)}")
        return f"分析过程中出现错误: {str(e)}"


def generate_analysis_report(filename: str, output_dir: str = "sjfx") -> str:
    """
    生成完整的数据分析报告（包含图表和MD文件）
    
    Args:
        filename: CSV文件名
        output_dir: 输出目录
    
    Returns:
        str: 报告文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文本分析
        text_analysis = _generate_text_analysis(filename)
        
        # 生成图表
        chart_files = generate_charts_for_analysis(filename, output_dir)
        
        # 生成MD报告
        report_content = _generate_md_report(filename, text_analysis, chart_files)
        
        # 保存MD文件
        report_file = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"分析报告已生成: {report_file}")
        return report_file
        
    except Exception as e:
        logging.error(f"生成分析报告失败: {str(e)}")
        return ""


def _generate_md_report(filename: str, text_analysis: str, chart_files: List[str]) -> str:
    """生成Markdown格式的分析报告"""
    report = f"""# 数据分析报告

## 数据文件信息
- **文件名**: {os.path.basename(filename)}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据行数**: {len(_safe_read_csv(filename))}

## 分析结果

{text_analysis}

## 可视化图表

"""
    
    # 添加图表
    if chart_files:
        for chart_file in chart_files:
            chart_name = os.path.basename(chart_file)
            report += f"### {chart_name.replace('.png', '').replace('_', ' ').title()}\n\n"
            report += f"![{chart_name}]({chart_name})\n\n"
    else:
        report += "暂无图表生成。\n\n"
    
    report += """## 总结

本报告基于提供的数据文件进行了全面的分析，包括数据概览、趋势分析、异常检测和可视化展示。如需更详细的分析或有其他问题，请随时联系。

---
*报告由数据分析系统自动生成*
"""
    
    return report


@tool
def generate_pdf_report(filename: str, output_dir: str = "test_output", report_title: str = "数据分析报告") -> str:
    """
    生成PDF格式的数据分析报告
    
    Args:
        filename: CSV文件名
        output_dir: 输出目录
        report_title: 报告标题
    
    Returns:
        str: PDF文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文本分析
        text_analysis = _generate_text_analysis(filename)
        
        # 生成图表
        chart_files = generate_charts_for_analysis(filename, output_dir)
        
        # 生成PDF文件
        pdf_file = _generate_pdf_with_reportlab(filename, text_analysis, chart_files, output_dir, report_title)
        
        return pdf_file
        
    except Exception as e:
        logging.error(f"生成PDF报告失败: {str(e)}")
        return f"❌ 生成PDF报告失败: {str(e)}"


@tool
def generate_html_to_pdf_report(filename: str, output_dir: str = "test_output", report_title: str = "数据分析报告") -> str:
    """
    使用HTML+CSS生成PDF格式的数据分析报告（支持更丰富的样式）
    
    Args:
        filename: CSV文件名
        output_dir: 输出目录
        report_title: 报告标题
    
    Returns:
        str: PDF文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文本分析
        text_analysis = _generate_text_analysis(filename)
        
        # 生成图表
        chart_files = generate_charts_for_analysis(filename, output_dir)
        
        # 生成HTML内容
        html_content = _generate_html_report(filename, text_analysis, chart_files, report_title)
        
        # 转换为PDF
        pdf_file = _convert_html_to_pdf(html_content, output_dir, report_title)
        
        return pdf_file
        
    except Exception as e:
        logging.error(f"生成HTML转PDF报告失败: {str(e)}")
        return f"❌ 生成HTML转PDF报告失败: {str(e)}"


def _generate_pdf_with_reportlab(filename: str, text_analysis: str, chart_files: List[str], output_dir: str, report_title: str) -> str:
    """使用ReportLab生成PDF报告"""
    try:
        # 创建PDF文件
        pdf_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # 创建PDF文档
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # 获取样式
        styles = getSampleStyleSheet()
        
        # 创建自定义样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # 添加标题
        story.append(Paragraph(report_title, title_style))
        story.append(Spacer(1, 20))
        
        # 添加文件信息
        story.append(Paragraph("数据文件信息", heading_style))
        file_info = f"""
        <b>文件名:</b> {os.path.basename(filename)}<br/>
        <b>分析时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>数据行数:</b> {len(_safe_read_csv(filename))}
        """
        story.append(Paragraph(file_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 添加分析结果
        story.append(Paragraph("分析结果", heading_style))
        
        # 将分析结果分段添加到PDF
        analysis_lines = text_analysis.split('\n')
        for line in analysis_lines:
            if line.strip():
                if line.startswith('📊') or line.startswith('📈') or line.startswith('📉'):
                    # 重要数据行，使用粗体
                    story.append(Paragraph(f"<b>{line}</b>", styles['Normal']))
                elif line.startswith('=') or line.startswith('-'):
                    # 分隔线，跳过
                    continue
                else:
                    story.append(Paragraph(line, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # 添加图表信息
        if chart_files:
            story.append(Paragraph("可视化图表", heading_style))
            for chart_file in chart_files:
                if os.path.exists(chart_file):
                    # 添加图表标题
                    chart_name = os.path.basename(chart_file).replace('.png', '').replace('_', ' ').title()
                    story.append(Paragraph(f"<b>{chart_name}</b>", styles['Normal']))
                    
                    # 添加图表图片
                    try:
                        img = Image(chart_file, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        story.append(Paragraph(f"图表加载失败: {str(e)}", styles['Normal']))
        
        # 添加总结
        story.append(Spacer(1, 20))
        story.append(Paragraph("总结", heading_style))
        summary_text = """
        本报告基于提供的数据文件进行了全面的分析，包括数据概览、趋势分析、异常检测和可视化展示。
        如需更详细的分析或有其他问题，请随时联系。
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        
        # 构建PDF
        doc.build(story)
        
        logging.info(f"PDF报告已生成: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logging.error(f"使用ReportLab生成PDF失败: {str(e)}")
        raise


def _generate_html_report(filename: str, text_analysis: str, chart_files: List[str], report_title: str) -> str:
    """生成HTML格式的报告内容"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            .file-info {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .analysis-content {{
                white-space: pre-line;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
            }}
            .chart-container {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .chart-title {{
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 16px;
            }}
            .summary {{
                background: #e8f5e8;
                padding: 20px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin-top: 30px;
            }}
            .footer {{
                text-align: center;
                color: #7f8c8d;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>
            
            <div class="file-info">
                <h2>数据文件信息</h2>
                <p><strong>文件名:</strong> {os.path.basename(filename)}</p>
                <p><strong>分析时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>数据行数:</strong> {len(_safe_read_csv(filename))}</p>
            </div>
            
            <h2>分析结果</h2>
            <div class="analysis-content">{text_analysis}</div>
            
            <h2>可视化图表</h2>
    """
    
    # 添加图表
    if chart_files:
        for chart_file in chart_files:
            if os.path.exists(chart_file):
                chart_name = os.path.basename(chart_file).replace('.png', '').replace('_', ' ').title()
                html_content += f"""
                <div class="chart-container">
                    <div class="chart-title">{chart_name}</div>
                    <img src="{os.path.basename(chart_file)}" alt="{chart_name}">
                </div>
                """
    else:
        html_content += "<p>暂无图表生成。</p>"
    
    html_content += """
            <div class="summary">
                <h2>总结</h2>
                <p>本报告基于提供的数据文件进行了全面的分析，包括数据概览、趋势分析、异常检测和可视化展示。如需更详细的分析或有其他问题，请随时联系。</p>
            </div>
            
            <div class="footer">
                <p>报告由数据分析系统自动生成</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def _convert_html_to_pdf(html_content: str, output_dir: str, report_title: str) -> str:
    """将HTML内容转换为PDF"""
    try:
        # 创建HTML文件
        html_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path = os.path.join(output_dir, html_filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 转换为PDF
        pdf_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # 使用WeasyPrint转换
        HTML(string=html_content).write_pdf(pdf_path)
        
        logging.info(f"HTML转PDF报告已生成: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logging.error(f"HTML转PDF失败: {str(e)}")
        raise

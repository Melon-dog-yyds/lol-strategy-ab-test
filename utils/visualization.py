"""
可视化工具模块
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


def create_win_rate_comparison_plot(win_rate_a: float, win_rate_b: float, 
                                   name_a: str, name_b: str) -> go.Figure:
    """
    创建胜率对比的交互式图表
    """
    fig = go.Figure()
    
    # 添加胜率柱状图
    fig.add_trace(go.Bar(
        x=[name_a, name_b],
        y=[win_rate_a * 100, win_rate_b * 100],
        text=[f'{win_rate_a*100:.2f}%', f'{win_rate_b*100:.2f}%'],
        textposition='auto',
        marker_color=['#4285F4', '#34A853'],
        opacity=0.8
    ))
    
    # 更新布局
    fig.update_layout(
        title='策略胜率对比',
        xaxis_title='策略',
        yaxis_title='胜率 (%)',
        showlegend=False,
        template='plotly_white',
        height=400
    )
    
    return fig


def create_confidence_interval_plot(diff: float, ci_lower: float, ci_upper: float, 
                                   alpha: float = 0.05) -> go.Figure:
    """
    创建置信区间图
    """
    fig = go.Figure()
    
    # 添加置信区间
    fig.add_trace(go.Scatter(
        x=[diff, diff],
        y=[0, 1],
        mode='lines+markers',
        line=dict(color='#EA4335', width=3),
        marker=dict(size=12),
        name='点估计'
    ))
    
    # 添加置信区间条
    fig.add_trace(go.Scatter(
        x=[ci_lower, ci_upper],
        y=[0.5, 0.5],
        mode='lines',
        line=dict(color='gray', width=8),
        name=f'{(1-alpha)*100:.0f}% 置信区间'
    ))
    
    # 添加0参考线
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=0, y1=1,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    # 更新布局
    fig.update_layout(
        title='胜率差异的置信区间',
        xaxis_title='胜率差异',
        yaxis=dict(showticklabels=False),
        showlegend=True,
        template='plotly_white',
        height=300,
        xaxis=dict(range=[min(ci_lower, -0.1), max(ci_upper, 0.1)])
    )
    
    return fig


def create_sample_size_plot(n_a: int, n_b: int, name_a: str, name_b: str) -> go.Figure:
    """
    创建样本量对比图
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[name_a, name_b],
        y=[n_a, n_b],
        text=[f'{n_a:,}', f'{n_b:,}'],
        textposition='auto',
        marker_color=['#FBBC05', '#EA4335'],
        opacity=0.8
    ))
    
    fig.update_layout(
        title='样本量对比',
        xaxis_title='策略',
        yaxis_title='样本量',
        showlegend=False,
        template='plotly_white',
        height=400
    )
    
    return fig


def create_power_curve_plot(effect_sizes: List[float], powers: List[float], 
                           current_effect: float, current_power: float) -> go.Figure:
    """
    创建功效曲线图
    """
    fig = go.Figure()
    
    # 添加功效曲线
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=powers,
        mode='lines',
        line=dict(color='#1E88E5', width=3),
        name='功效曲线'
    ))
    
    # 添加当前效应量和功效点
    fig.add_trace(go.Scatter(
        x=[current_effect],
        y=[current_power],
        mode='markers',
        marker=dict(size=15, color='#EA4335'),
        name='当前状态'
    ))
    
    # 添加80%功效参考线
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                 annotation_text="80% 功效目标")
    
    fig.update_layout(
        title='统计功效曲线',
        xaxis_title='效应量 (Cohen\'s h)',
        yaxis_title='统计功效 (1-β)',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig
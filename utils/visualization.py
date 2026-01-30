"""
可视化工具模块 - 使用Plotly（更好的中文支持）
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def create_chinese_font_family():
    """返回中文字体族"""
    return "'Noto Sans SC', 'Microsoft YaHei', sans-serif"


def create_win_rate_bar_chart(win_rate_a: float, win_rate_b: float, 
                             name_a: str, name_b: str) -> go.Figure:
    """
    创建胜率对比柱状图（Plotly版本）
    """
    # 准备数据
    strategies = [name_a, name_b]
    win_rates = [win_rate_a * 100, win_rate_b * 100]
    
    # 创建图表
    fig = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=win_rates,
            text=[f'{rate:.2f}%' for rate in win_rates],
            textposition='auto',
            marker_color=['#4285F4', '#34A853'],
            opacity=0.8,
            textfont=dict(size=14, color='white')
        )
    ])
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='策略胜率对比',
            font=dict(size=20, family=create_chinese_font_family()),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='策略',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=14, family=create_chinese_font_family())
        ),
        yaxis=dict(
            title='胜率 (%)',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=12)
        ),
        showlegend=False,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_sample_size_chart(n_a: int, n_b: int, name_a: str, name_b: str) -> go.Figure:
    """
    创建样本量对比图（Plotly版本）
    """
    fig = go.Figure(data=[
        go.Bar(
            x=[name_a, name_b],
            y=[n_a, n_b],
            text=[f'{n_a:,}', f'{n_b:,}'],
            textposition='auto',
            marker_color=['#FBBC05', '#EA4335'],
            opacity=0.8,
            textfont=dict(size=14, color='white')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='样本量对比',
            font=dict(size=20, family=create_chinese_font_family()),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='策略',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=14, family=create_chinese_font_family())
        ),
        yaxis=dict(
            title='样本量',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=12)
        ),
        showlegend=False,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def create_confidence_interval_plot(diff: float, ci_lower: float, ci_upper: float,
                                   name_a: str, name_b: str, alpha: float = 0.05) -> go.Figure:
    """
    创建置信区间图（Plotly版本）
    """
    fig = go.Figure()
    
    # 添加置信区间条
    fig.add_trace(go.Scatter(
        x=[ci_lower, ci_upper],
        y=['区间', '区间'],
        mode='lines',
        line=dict(color='gray', width=10),
        name=f'{(1-alpha)*100:.0f}% 置信区间',
        showlegend=True
    ))
    
    # 添加点估计
    fig.add_trace(go.Scatter(
        x=[diff],
        y=['区间'],
        mode='markers',
        marker=dict(size=20, color='#EA4335'),
        name='点估计',
        showlegend=True,
        hovertemplate=f'差异: {diff:.3f}<extra></extra>'
    ))
    
    # 添加0参考线
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    # 计算胜率差异百分比
    diff_percent = diff * 100
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    
    fig.update_layout(
        title=dict(
            text=f'胜率差异: {diff_percent:+.2f}% ({name_b} - {name_a})',
            font=dict(size=18, family=create_chinese_font_family()),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=f'胜率差异 (%)',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=12),
            range=[min(ci_lower_percent - 5, -5), max(ci_upper_percent + 5, 5)]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        showlegend=True,
        template='plotly_white',
        height=300,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(family=create_chinese_font_family())
        )
    )
    
    # 添加注释
    fig.add_annotation(
        x=ci_lower_percent,
        y=0.2,
        text=f"{ci_lower_percent:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray",
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=ci_upper_percent,
        y=0.2,
        text=f"{ci_upper_percent:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray",
        font=dict(size=12)
    )
    
    return fig


def create_power_analysis_chart(current_power: float, required_n: int, 
                               effect_size: float) -> go.Figure:
    """
    创建功效分析图
    """
    # 创建模拟数据
    sample_sizes = list(range(50, 1001, 50))
    powers = []
    
    # 简化功效计算
    for n in sample_sizes:
        # 简化公式：功效随样本量增加而增加
        power = min(0.99, 0.8 * (n / required_n) ** 0.5)
        powers.append(power)
    
    fig = go.Figure()
    
    # 添加功效曲线
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=powers,
        mode='lines',
        line=dict(color='#1E88E5', width=3),
        name='功效曲线',
        hovertemplate='样本量: %{x}<br>功效: %{y:.2%}<extra></extra>'
    ))
    
    # 添加当前状态点
    fig.add_trace(go.Scatter(
        x=[required_n],
        y=[0.8],
        mode='markers+text',
        marker=dict(size=15, color='green'),
        name='目标样本量',
        text=['目标'],
        textposition='top center',
        hovertemplate=f'目标样本量: {required_n:,}<br>目标功效: 80%<extra></extra>'
    ))
    
    # 添加80%功效线
    fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                 annotation_text="80% 功效目标",
                 annotation_font=dict(color="green"),
                 annotation_position="bottom right")
    
    fig.update_layout(
        title=dict(
            text='统计功效分析',
            font=dict(size=20, family=create_chinese_font_family()),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='样本量（每组）',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='统计功效 (1-β)',
            title_font=dict(size=16, family=create_chinese_font_family()),
            tickfont=dict(size=12),
            tickformat='.0%'
        ),
        showlegend=True,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(family=create_chinese_font_family())
        )
    )
    
    return fig
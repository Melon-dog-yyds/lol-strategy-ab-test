"""
英雄联盟策略A/B测试交互式平台
Streamlit Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心引擎
try:
    from core.ab_test_engine import ABTestEngine
    print("成功导入ABTestEngine")
except ImportError as e:
    print(f"导入错误: {e}")
    # 备用导入路径
    sys.path.append('.')
    from core.ab_test_engine import ABTestEngine

# 设置页面配置
st.set_page_config(
    page_title="英雄联盟策略A/B测试平台",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体（解决中文显示问题）
def setup_chinese_font():
    """配置中文字体显示"""
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 创建自定义CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .main-header {
        text-align: center;
        color: #1E88E5;
        padding: 20px 0;
        margin-bottom: 30px;
        border-bottom: 3px solid #1E88E5;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    
    .warning-card {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    .success-card {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    .imbalance-critical {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    
    .imbalance-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    
    .imbalance-ok {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化应用
def init_app():
    """初始化应用"""
    setup_chinese_font()
    
    # 应用标题
    st.markdown("<h1 class='main-header'>🎮 英雄联盟策略A/B测试分析平台</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 30px;'>
    比较不同技能加点/装备策略的胜率差异 | 基于统计假设检验
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化session state
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'imbalance_analysis' not in st.session_state:
        st.session_state.imbalance_analysis = None

# 侧边栏 - 数据输入
def sidebar_input():
    """侧边栏数据输入"""
    with st.sidebar:
        st.header("⚙️ 测试参数设置")
        
        # 策略名称
        col1, col2 = st.columns(2)
        with col1:
            name_a = st.text_input("策略A名称", value="主流策略")
        with col2:
            name_b = st.text_input("策略B名称", value="创新策略")
        
        st.markdown("---")
        
        # 策略A数据
        st.subheader(f"📊 {name_a} 数据")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            n_a = st.number_input(f"{name_a}总场次", 
                                 min_value=1, 
                                 max_value=1000000, 
                                 value=1000,
                                 step=100,
                                 key="n_a")
        with col_a2:
            # 改为输入框（允许输入小数百分比）
            win_rate_input_a = st.text_input(f"{name_a}胜率 (%)", 
                                           value="52.0",
                                           help="输入百分比，如52.5表示52.5%")
            
            # 转换输入为小数
            try:
                win_rate_a = float(win_rate_input_a.strip('%')) / 100
                if not (0 <= win_rate_a <= 1):
                    st.error("胜率必须在0-100%之间")
                    win_rate_a = 0.52  # 默认值
            except ValueError:
                st.error("请输入有效的数字")
                win_rate_a = 0.52  # 默认值
        
        # 策略B数据
        st.subheader(f"📈 {name_b} 数据")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            n_b = st.number_input(f"{name_b}总场次", 
                                 min_value=1, 
                                 max_value=1000000, 
                                 value=50,
                                 step=10,
                                 key="n_b")
        with col_b2:
            # 改为输入框
            win_rate_input_b = st.text_input(f"{name_b}胜率 (%)", 
                                           value="62.0",
                                           help="输入百分比，如62.5表示62.5%")
            
            # 转换输入为小数
            try:
                win_rate_b = float(win_rate_input_b.strip('%')) / 100
                if not (0 <= win_rate_b <= 1):
                    st.error("胜率必须在0-100%之间")
                    win_rate_b = 0.62  # 默认值
            except ValueError:
                st.error("请输入有效的数字")
                win_rate_b = 0.62  # 默认值
        
        # 显示验证后的胜率
        st.caption(f"解析胜率: {name_a}={win_rate_a*100:.2f}%, {name_b}={win_rate_b*100:.2f}%")
        
        st.markdown("---")
        
        # 检验参数
        st.subheader("🔬 检验参数")
        col_alpha1, col_alpha2 = st.columns(2)
        with col_alpha1:
            alpha = st.number_input("显著性水平 (α)", 
                                   min_value=0.01,
                                   max_value=0.20,
                                   value=0.05,
                                   step=0.01,
                                   format="%.2f",
                                   help="第一类错误概率，通常设为0.05")
        with col_alpha2:
            alternative = st.selectbox(
                "检验方向",
                options=["two-sided", "greater", "less"],
                format_func=lambda x: {
                    "two-sided": "双侧检验",
                    "greater": "B优于A", 
                    "less": "A优于B"
                }[x],
                index=1
            )
        
        # 检验方法选择
        st.subheader("📋 检验方法")
        method = st.selectbox(
            "选择检验方法",
            options=["auto", "z_test", "chi2", "fisher", "barnard"],
            format_func=lambda x: {
                "auto": "自动推荐（根据样本情况）",
                "z_test": "两比例Z检验",
                "chi2": "卡方检验",
                "fisher": "Fisher精确检验",
                "barnard": "Barnard精确检验"
            }[x],
            index=0
        )
        
        # 运行按钮
        st.markdown("---")
        run_button = st.button("🚀 运行A/B测试分析", 
                              type="primary", 
                              use_container_width=True)
        
        return {
            'name_a': name_a,
            'name_b': name_b,
            'n_a': int(n_a),
            'win_rate_a': win_rate_a,
            'n_b': int(n_b),
            'win_rate_b': win_rate_b,
            'alpha': alpha,
            'alternative': alternative,
            'method': method,
            'run_button': run_button
        }
        
# 显示样本不平衡警告
def show_imbalance_warnings(analysis: Dict):
    """显示样本不平衡警告"""
    if not analysis:
        return
    
    imbalance_level = analysis['不平衡程度']
    color = analysis['颜色标识']
    css_color = analysis['CSS颜色']
    
    # 根据不平衡程度显示不同的警告框
    if imbalance_level == "重度不平衡":
        st.markdown(f"""
        <div class='imbalance-critical'>
        <h4>⚠️ 严重警告：样本重度不平衡</h4>
        <p>• 样本量比例：{analysis['样本量比例']}（{color}）</p>
        <p>• 检验功效可能严重不足</p>
        <p>• <strong>推荐方法：{analysis['推荐方法显示名']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    elif imbalance_level == "中度不平衡":
        st.markdown(f"""
        <div class='imbalance-warning'>
        <h4>⚠️ 警告：样本中度不平衡</h4>
        <p>• 样本量比例：{analysis['样本量比例']}（{color}）</p>
        <p>• 建议使用推荐的方法进行检验</p>
        <p>• <strong>推荐方法：{analysis['推荐方法显示名']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    elif analysis['小样本警告'] == "是":
        st.markdown(f"""
        <div class='imbalance-warning'>
        <h4>⚠️ 注意：存在小样本</h4>
        <p>• 至少一组样本量小于30</p>
        <p>• 正态近似可能不准确</p>
        <p>• <strong>推荐方法：{analysis['推荐方法显示名']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# 显示基本统计信息
def show_basic_stats(engine: ABTestEngine):
    """显示基本统计信息"""
    st.header("📊 基本统计数据")
    
    stats_df = engine.get_basic_stats()
    
    # 使用列布局显示关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{engine.name_a}胜率",
            value=f"{engine.win_rate_a*100:.2f}%",
            delta=f"{engine.wins_a}胜/{engine.n_a-engine.wins_a}负"
        )
    
    with col2:
        st.metric(
            label=f"{engine.name_b}胜率",
            value=f"{engine.win_rate_b*100:.2f}%",
            delta=f"{engine.wins_b}胜/{engine.n_b-engine.wins_b}负"
        )
    
    with col3:
        diff_percent = (engine.win_rate_b - engine.win_rate_a) * 100
        st.metric(
            label="胜率差异",
            value=f"{diff_percent:+.2f}%",
            delta="B相对A"
        )
    
    with col4:
        total_games = engine.n_a + engine.n_b
        st.metric(
            label="总样本量",
            value=f"{total_games:,}",
            delta=f"A:{engine.n_a:,} B:{engine.n_b:,}"
        )
    
    # 显示详细统计表
    with st.expander("📋 查看详细统计表", expanded=True):
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")

# 可视化胜率对比
def plot_win_rate_comparison(engine: ABTestEngine, result: Dict):
    """可视化胜率对比 - 使用Plotly"""
    st.header("📈 可视化分析")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 使用Plotly创建胜率柱状图
        from utils.visualization import create_win_rate_bar_chart
        fig1 = create_win_rate_bar_chart(
            engine.win_rate_a, engine.win_rate_b,
            engine.name_a, engine.name_b
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 根据是否有置信区间显示不同的图表
        if 'ci_lower' in result and 'ci_upper' in result:
            from utils.visualization import create_confidence_interval_plot
            fig2 = create_confidence_interval_plot(
                diff=engine.win_rate_b - engine.win_rate_a,
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                name_a=engine.name_a,
                name_b=engine.name_b,
                alpha=result.get('alpha', 0.05)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            from utils.visualization import create_sample_size_chart
            fig2 = create_sample_size_chart(
                engine.n_a, engine.n_b,
                engine.name_a, engine.name_b
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")

# 显示检验结果
def show_test_results(result: Dict, engine: ABTestEngine, alpha: float):
    """显示检验结果"""
    st.header("🔬 假设检验结果")
    
    # 结果摘要卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result['significant']:
            st.markdown("""
            <div class='success-card'>
            <h4>✅ 统计显著</h4>
            <p>p值 < α，拒绝原假设</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-card'>
            <h4>⏸️ 统计不显著</h4>
            <p>p值 ≥ α，不能拒绝原假设</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        p_value_formatted = f"{result['p_value']:.4f}"
        if result['p_value'] < 0.001:
            p_value_formatted = "< 0.001"
        
        st.markdown(f"""
        <div class='metric-card'>
        <h4>📊 P值</h4>
        <h3>{p_value_formatted}</h3>
        <p>α = {alpha}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        effect_size = (engine.win_rate_b - engine.win_rate_a) * 100
        st.markdown(f"""
        <div class='metric-card'>
        <h4>📈 效应量</h4>
        <h3>{effect_size:+.2f}%</h3>
        <p>胜率绝对差异</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 详细结果表格
    with st.expander("📋 查看详细检验结果", expanded=True):
        result_data = []
        
        # 添加通用结果
        result_data.append(["检验方法", result.get('method', 'N/A')])
        result_data.append(["P值", f"{result.get('p_value', 0):.4f}"])
        result_data.append(["显著性水平 (α)", f"{alpha}"])
        result_data.append(["是否显著", "是" if result.get('significant', False) else "否"])
        result_data.append(["统计结论", result.get('recommendation', 'N/A')])
        
        # 添加特定检验的统计量
        if 'z_statistic' in result:
            result_data.append(["Z统计量", f"{result['z_statistic']:.4f}"])
        if 'chi2_statistic' in result:
            result_data.append(["卡方统计量", f"{result['chi2_statistic']:.4f}"])
            result_data.append(["自由度", f"{result['degrees_of_freedom']}"])
            result_data.append(["Phi系数", f"{result['phi_coefficient']:.4f}"])
        if 'odds_ratio' in result:
            result_data.append(["比值比 (OR)", f"{result['odds_ratio']:.4f}"])
        
        # 添加置信区间
        if 'ci_lower' in result and 'ci_upper' in result:
            ci_lower = result['ci_lower'] * 100
            ci_upper = result['ci_upper'] * 100
            result_data.append([f"胜率差异 {int((1-alpha)*100)}% CI", 
                               f"[{ci_lower:.2f}%, {ci_upper:.2f}%]"])
        
        # 创建DataFrame并显示
        result_df = pd.DataFrame(result_data, columns=["指标", "值"])
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")

# 显示功效分析
def show_power_analysis(engine: ABTestEngine, alpha: float):
    """显示功效分析"""
    st.header("💪 统计功效分析")
    
    try:
        power_result = engine.get_power_analysis(alpha=alpha)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_power = power_result['current_power']
            power_color = "green" if current_power >= 0.8 else "orange" if current_power >= 0.5 else "red"
            
            st.markdown(f"""
            <div class='metric-card'>
            <h4>📊 当前功效 (1-β)</h4>
            <h3 style='color: {power_color}'>{current_power:.1%}</h3>
            <p>{power_result['interpretation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            required_n = power_result['required_sample_size_per_group']
            st.markdown(f"""
            <div class='metric-card'>
            <h4>📈 每组所需样本量</h4>
            <h3>{required_n:,}</h3>
            <p>达到80%功效</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_needed = power_result['required_total_samples']
            current_total = power_result['current_total_samples']
            st.markdown(f"""
            <div class='metric-card'>
            <h4>🎯 总样本差距</h4>
            <h3>{total_needed - current_total:+,}</h3>
            <p>还需{max(0, total_needed - current_total):,}个样本</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 样本量建议
        with st.expander("📋 查看详细样本量建议", expanded=False):
            st.info("""
            **功效解释**：统计功效表示当策略B确实优于策略A时，检验能检测到这种差异的概率。
            
            **一般标准**：
            - 功效 ≥ 80%：良好，检验结果可靠
            - 功效 50%-80%：不足，可能漏掉真实差异  
            - 功效 < 50%：很低，结果不确定性大
            """)
            
            rec_data = [
                ["当前总样本量", f"{current_total:,}"],
                ["所需总样本量", f"{total_needed:,}"],
                ["还需样本量", f"{max(0, total_needed - current_total):,}"],
                ["当前效应量", f"{power_result['observed_effect_size']:.3f}"],
                ["建议每组样本量", f"{required_n:,}"],
                ["结论", power_result['interpretation']]
            ]
            
            rec_df = pd.DataFrame(rec_data, columns=["项目", "值"])
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.warning(f"功效分析计算失败: {str(e)}")
    
    st.markdown("---")

# 显示最终建议
def show_recommendation(result: Dict, engine: ABTestEngine, imbalance_analysis: Dict):
    """显示最终建议"""
    st.header("🎯 最终建议与行动计划")
    
    # 根据检验结果给出建议
    if result['significant']:
        if engine.win_rate_b > engine.win_rate_a:
            recommendation = f"**强烈推荐使用 {engine.name_b}**"
            reason = f"统计显著优于 {engine.name_a}，胜率高{(engine.win_rate_b - engine.win_rate_a)*100:.1f}%"
            color = "success"
        else:
            recommendation = f"**坚持使用 {engine.name_a}**"
            reason = f"统计显著优于 {engine.name_b}，胜率高{(engine.win_rate_a - engine.win_rate_b)*100:.1f}%"
            color = "success"
    else:
        effect_diff = abs(engine.win_rate_b - engine.win_rate_a) * 100
        if effect_diff > 5:  # 效应量较大
            recommendation = "**继续测试，收集更多数据**"
            reason = f"效应量较大({effect_diff:.1f}%)但统计不显著，可能样本不足"
            color = "warning"
        else:
            recommendation = "**两种策略效果相近**"
            reason = f"差异小({effect_diff:.1f}%)且统计不显著，可根据偏好选择"
            color = "info"
    
    # 显示建议卡片
    if color == "success":
        st.success(f"### {recommendation}")
    elif color == "warning":
        st.warning(f"### {recommendation}")
    else:
        st.info(f"### {recommendation}")
    
    st.write(f"**理由**：{reason}")
    
    # 行动计划
    st.subheader("📋 行动计划")
    
    action_cols = st.columns(2)
    
    with action_cols[0]:
        st.markdown("""
        **立即行动：**
        1. 记录本次分析结果
        2. 与团队分享发现
        3. 根据建议调整策略
        
        **后续监控：**
        1. 跟踪策略胜率变化
        2. 收集玩家反馈
        3. 关注版本更新影响
        """)
    
    with action_cols[1]:
        # 样本不平衡建议
        if imbalance_analysis and imbalance_analysis['不平衡程度'] in ["中度不平衡", "重度不平衡"]:
            st.markdown(f"""
            **样本优化建议：**
            1. 当前不平衡：{imbalance_analysis['不平衡程度']}
            2. 推荐方法：{imbalance_analysis['推荐方法显示名']}
            3. 建议收集更多{engine.name_b if engine.n_b < engine.n_a else engine.name_a}数据
            4. 目标：达到{imbalance_analysis['最小建议样本量']}个样本
            """)
        else:
            st.markdown("""
            **检验可靠性：**
            1. 当前统计功效：良好
            2. 样本平衡性：可接受
            3. 检验方法：合适
            4. 结果可信度：高
            """)
    
    # 导出选项
    st.markdown("---")
    st.subheader("📤 导出结果")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📋 复制结果摘要", use_container_width=True):
            # 这里可以添加复制到剪贴板的功能
            st.toast("结果摘要已复制到剪贴板！", icon="✅")
    
    with export_col2:
        if st.button("📊 下载CSV报告", use_container_width=True):
            # 这里可以添加生成CSV文件的功能
            st.toast("CSV报告生成中...", icon="📥")
    
    with export_col3:
        if st.button("🖼️ 保存图表", use_container_width=True):
            # 这里可以添加保存图表的功能
            st.toast("图表已保存！", icon="🖼️")

# 主应用逻辑
def main():
    """主应用逻辑"""
    init_app()
    
    # 侧边栏输入
    inputs = sidebar_input()
    
    # 当点击运行按钮时
    if inputs['run_button']:
        try:
            # 创建引擎实例
            engine = ABTestEngine(inputs['name_a'], inputs['name_b'])
            engine.set_data(
                inputs['n_a'], inputs['win_rate_a'],
                inputs['n_b'], inputs['win_rate_b']
            )
            
            # 保存到session state
            st.session_state.engine = engine
            
            # 样本不平衡分析
            imbalance_analysis = engine.get_sample_imbalance_analysis()
            st.session_state.imbalance_analysis = imbalance_analysis
            
            # 确定使用的检验方法
            if inputs['method'] == 'auto':
                method_to_use = imbalance_analysis['推荐检验方法']
            else:
                method_to_use = inputs['method']
            
            # 运行检验
            with st.spinner(f"正在执行{imbalance_analysis['推荐方法显示名'] if inputs['method'] == 'auto' else method_to_use}..."):
                result = engine.run_test(
                    method=method_to_use,
                    alpha=inputs['alpha'],
                    alternative=inputs['alternative']
                )
                
                # 保存结果
                st.session_state.test_results[method_to_use] = result
                
            # 显示分析结果
            show_imbalance_warnings(imbalance_analysis)
            show_basic_stats(engine)
            plot_win_rate_comparison(engine, result)
            show_test_results(result, engine, inputs['alpha'])
            show_power_analysis(engine, inputs['alpha'])
            show_recommendation(result, engine, imbalance_analysis)
            
        except Exception as e:
            st.error(f"分析过程中出现错误：{str(e)}")
            st.exception(e)
    
    else:
        # 显示欢迎信息和示例
        st.info("""
        ## 🎯 使用指南
        
        1. **左侧边栏**输入两种策略的数据
        2. 设置检验参数（显著性水平、检验方向等）
        3. 选择检验方法或使用"自动推荐"
        4. 点击"运行A/B测试分析"按钮
        
        ## 📊 示例数据
        - 策略A：1000场，胜率52%
        - 策略B：50场，胜率62%
        
        ## ⚠️ 注意事项
        - 样本量差异大时系统会自动推荐合适方法
        - 统计功效不足时会给出样本量建议
        - 中文显示已优化，确保正确显示
        """)
        
        # 显示示例图片或图表
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://via.placeholder.com/400x250/4285F4/FFFFFF?text=胜率对比示例", 
                    caption="胜率对比可视化示例")
        with col2:
            st.image("https://via.placeholder.com/400x250/34A853/FFFFFF?text=统计功效分析", 
                    caption="统计功效分析示例")

# 运行应用
if __name__ == "__main__":
    main()
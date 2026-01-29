"""
A/B测试统计引擎 - 英雄联盟加点策略分析
支持多种统计检验方法，处理样本不平衡情况
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class ABTestEngine:
    """A/B测试统计引擎"""
    
    def __init__(self, name_a: str = "策略A", name_b: str = "策略B"):
        """
        初始化A/B测试引擎
        
        参数:
        ----------
        name_a : str
            策略A的名称
        name_b : str
            策略B的名称
        """
        self.name_a = name_a
        self.name_b = name_b
        
        # 存储输入数据
        self.n_a = None
        self.n_b = None
        self.win_rate_a = None
        self.win_rate_b = None
        self.wins_a = None
        self.wins_b = None
        
        # 存储结果
        self.results = {}
        
    def set_data(self, n_a: int, win_rate_a: float, n_b: int, win_rate_b: float):
        """
        设置测试数据
        
        参数:
        ----------
        n_a : int
            策略A的总场次数
        win_rate_a : float
            策略A的胜率（0-1之间）
        n_b : int
            策略B的总场次数
        win_rate_b : float
            策略B的胜率（0-1之间）
        """
        # 验证数据
        self._validate_data(n_a, win_rate_a, n_b, win_rate_b)
        
        # 存储数据
        self.n_a = n_a
        self.n_b = n_b
        self.win_rate_a = win_rate_a
        self.win_rate_b = win_rate_b
        self.wins_a = int(round(n_a * win_rate_a))
        self.wins_b = int(round(n_b * win_rate_b))
        
        # 打印确认
        print(f"数据设置成功:")
        print(f"  {self.name_a}: {n_a}场, 胜率{win_rate_a*100:.1f}%, 胜场{self.wins_a}")
        print(f"  {self.name_b}: {n_b}场, 胜率{win_rate_b*100:.1f}%, 胜场{self.wins_b}")
        
    def _validate_data(self, n_a: int, win_rate_a: float, n_b: int, win_rate_b: float):
        """验证输入数据的有效性"""
        # 检查场次数
        if n_a <= 0 or n_b <= 0:
            raise ValueError("场次数必须为正整数")
        
        # 检查胜率范围
        if not (0 <= win_rate_a <= 1) or not (0 <= win_rate_b <= 1):
            raise ValueError("胜率必须在0到1之间")
        
        # 检查样本量是否过小
        if n_a < 30 or n_b < 30:
            print("警告: 样本量较小(<30)，检验结果可能不可靠")
        
        # 检查胜场数是否为整数（允许轻微误差）
        if abs(n_a * win_rate_a - round(n_a * win_rate_a)) > 0.001:
            print(f"注意: {self.name_a}的胜场数({n_a * win_rate_a:.2f})不是整数，已四舍五入")
        if abs(n_b * win_rate_b - round(n_b * win_rate_b)) > 0.001:
            print(f"注意: {self.name_b}的胜场数({n_b * win_rate_b:.2f})不是整数，已四舍五入")
    
    def get_basic_stats(self) -> pd.DataFrame:
        """获取基本统计数据"""
        if any(v is None for v in [self.n_a, self.win_rate_a, self.n_b, self.win_rate_b]):
            raise ValueError("请先设置数据")
        
        # 计算各项统计量
        loss_a = self.n_a - self.wins_a
        loss_b = self.n_b - self.wins_b
        
        data = {
            '指标': ['总场次', '胜场数', '负场数', '胜率', '负率', '样本占比'],
            self.name_a: [
                f"{self.n_a:,}",
                f"{self.wins_a:,}",
                f"{loss_a:,}",
                f"{self.win_rate_a*100:.2f}%",
                f"{(1-self.win_rate_a)*100:.2f}%",
                f"{self.n_a/(self.n_a+self.n_b)*100:.1f}%"
            ],
            self.name_b: [
                f"{self.n_b:,}",
                f"{self.wins_b:,}",
                f"{loss_b:,}",
                f"{self.win_rate_b*100:.2f}%",
                f"{(1-self.win_rate_b)*100:.2f}%",
                f"{self.n_b/(self.n_a+self.n_b)*100:.1f}%"
            ],
            '绝对差值': [
                f"{self.n_b - self.n_a:+,}",
                f"{self.wins_b - self.wins_a:+,}",
                f"{loss_b - loss_a:+,}",
                f"{(self.win_rate_b - self.win_rate_a)*100:+.2f}%",
                f"{((1-self.win_rate_b) - (1-self.win_rate_a))*100:+.2f}%",
                f"{(self.n_b/(self.n_a+self.n_b) - self.n_a/(self.n_a+self.n_b))*100:+.1f}%"
            ]
        }
        
        return pd.DataFrame(data)

    def _get_recommended_method(self) -> str:
        """
        根据样本特征推荐最适合的检验方法
        """
        if any(v is None for v in [self.n_a, self.n_b, self.wins_a, self.wins_b]):
            return 'z_test'  # 默认
        
        # 计算样本量比
        ratio = min(self.n_a, self.n_b) / max(self.n_a, self.n_b) if max(self.n_a, self.n_b) > 0 else 0
        
        # 判断小样本
        is_small_sample = min(self.n_a, self.n_b) < 30
        
        # 判断极度不平衡
        is_extreme_imbalance = ratio < 0.1
        
        # 计算最小期望计数（用于卡方/Fisher选择）
        min_expected = min(
            self.wins_a, self.n_a - self.wins_a,
            self.wins_b, self.n_b - self.wins_b
        )
        
        # 推荐逻辑
        if is_small_sample or is_extreme_imbalance:
            if min_expected >= 5:
                return 'fisher'  # Fisher精确检验
            else:
                return 'barnard'  # Barnard检验（极端小样本）
        elif 0.1 <= ratio < 0.3:
            return 'z_test'  # Z检验（不合并方差）
        else:
            return 'chi2' if min_expected >= 5 else 'fisher'
    
    def get_sample_imbalance_analysis(self) -> Dict:
        """
        分析样本不平衡情况并给出建议
        """
        if any(v is None for v in [self.n_a, self.n_b]):
            raise ValueError("请先设置数据")
        
        # 计算不平衡指标
        total = self.n_a + self.n_b
        ratio = min(self.n_a, self.n_b) / max(self.n_a, self.n_b) if max(self.n_a, self.n_b) > 0 else 0
        
        # 判断标准
        if ratio >= 0.67:
            imbalance_level = "平衡"
            color = "🟢"
            css_color = "green"
        elif ratio >= 0.33:
            imbalance_level = "轻度不平衡"
            color = "🟡"
            css_color = "orange"
        elif ratio >= 0.1:
            imbalance_level = "中度不平衡"
            color = "🟠"
            css_color = "darkorange"
        else:
            imbalance_level = "重度不平衡"
            color = "🔴"
            css_color = "red"
        
        # 小样本判断
        is_small_sample = min(self.n_a, self.n_b) < 30
        
        # 推荐方法
        recommended_method = self._get_recommended_method()
        method_name_map = {
            'z_test': '两比例Z检验',
            'chi2': '卡方检验',
            'fisher': 'Fisher精确检验',
            'barnard': 'Barnard精确检验'
        }
        
        # 样本量建议
        min_recommended = max(50, int(0.3 * max(self.n_a, self.n_b)))
        
        # 解释和建议
        if imbalance_level == "重度不平衡":
            advice = [
                f"样本量极度不平衡（{color} {imbalance_level}）",
                f"• 检验功效可能严重不足",
                f"• 小样本组的结果不确定性很大",
                f"• 推荐使用：{method_name_map[recommended_method]}",
                f"• 建议至少收集{min_recommended}个样本到小样本组"
            ]
        elif is_small_sample:
            advice = [
                f"存在小样本问题（{color} {imbalance_level}）",
                f"• 至少一组样本量小于30",
                f"• 正态近似可能不成立",
                f"• 推荐使用：{method_name_map[recommended_method]}",
                f"• 置信区间可能较宽，解释需谨慎"
            ]
        else:
            advice = [
                f"样本情况可接受（{color} {imbalance_level}）",
                f"• 样本量比例：{ratio:.2%}",
                f"• 推荐使用：{method_name_map[recommended_method]}",
                f"• 大部分检验方法适用"
            ]
        
        return {
            '样本总量': total,
            '样本量A': self.n_a,
            '样本量B': self.n_b,
            '样本量比例': f"{ratio:.2%}",
            '不平衡程度': imbalance_level,
            '小样本警告': "是" if is_small_sample else "否",
            '推荐检验方法': recommended_method,
            '推荐方法显示名': method_name_map[recommended_method],
            '详细建议': advice,
            '颜色标识': color,
            'CSS颜色': css_color,
            '最小建议样本量': min_recommended
        }
    
    def get_sample_size_recommendation(self, 
                                      alpha: float = 0.05, 
                                      power: float = 0.8,
                                      effect_size: float = None) -> Dict:
        """
        获取样本量建议
        
        参数:
        ----------
        alpha : float
            显著性水平
        power : float
            期望的功效
        effect_size : float, optional
            期望检测的效应量，如果为None则使用当前观测效应量
        """
        if any(v is None for v in [self.win_rate_a, self.win_rate_b]):
            raise ValueError("请先设置数据")
        
        # 计算当前效应量（Cohen's h）
        if effect_size is None:
            h = 2 * np.arcsin(np.sqrt(self.win_rate_b)) - 2 * np.arcsin(np.sqrt(self.win_rate_a))
            effect_size = abs(h)
        
        # 计算所需样本量（基于比例检验）
        from statsmodels.stats.power import NormalIndPower
        analysis = NormalIndPower()
        
        # 使用当前比例作为参考
        ratio = self.n_b / self.n_a if self.n_a > 0 else 1
        
        # 计算达到指定功效所需样本量
        required_n = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio
        )
        
        required_n = int(np.ceil(required_n))
        
        # 基于当前不平衡程度的建议
        imbalance_ratio = min(self.n_a, self.n_b) / max(self.n_a, self.n_b) if max(self.n_a, self.n_b) > 0 else 0
        
        if imbalance_ratio < 0.2:
            recommendation = f"当前样本严重不平衡，建议按{required_n}:{required_n}的平衡设计收集数据"
            optimal_ratio = 1.0
        elif imbalance_ratio < 0.5:
            recommendation = f"当前样本中度不平衡，建议按{required_n}:{int(required_n*0.7)}的比例收集数据"
            optimal_ratio = 0.7
        else:
            recommendation = f"当前样本相对平衡，可按{required_n}:{int(required_n*ratio)}的比例收集数据"
            optimal_ratio = ratio
        
        return {
            '当前效应量': effect_size,
            '显著性水平': alpha,
            '目标功效': power,
            '当前样本比例': f"{ratio:.2f}",
            '每组建议样本量': required_n,
            '策略A建议样本量': required_n,
            '策略B建议样本量': int(required_n * optimal_ratio),
            '总建议样本量': required_n + int(required_n * optimal_ratio),
            '样本量建议': recommendation,
            '最优比例': optimal_ratio
        }
    
    def run_test(self, method: str = 'z_test', alpha: float = 0.05, 
                 alternative: str = 'two-sided') -> Dict:
        """
        执行假设检验
        
        参数:
        ----------
        method : str
            检验方法: 'z_test', 'chi2', 'fisher', 'barnard'
        alpha : float
            显著性水平，默认0.05
        alternative : str
            备择假设: 'two-sided', 'greater', 'less'
            
        返回:
        ----------
        Dict : 包含检验结果的字典
        """
        # 验证数据已设置
        if any(v is None for v in [self.n_a, self.win_rate_a, self.n_b, self.win_rate_b]):
            raise ValueError("请先设置数据")
        
        # 创建2x2列联表
        table = np.array([
            [self.wins_a, self.n_a - self.wins_a],  # 策略A: [胜, 负]
            [self.wins_b, self.n_b - self.wins_b]   # 策略B: [胜, 负]
        ])
        
        # 执行选择的检验
        if method == 'z_test':
            result = self._z_test(table, alpha, alternative)
        elif method == 'chi2':
            result = self._chi2_test(table, alpha)
        elif method == 'fisher':
            result = self._fisher_test(table, alpha, alternative)
        elif method == 'barnard':
            result = self._barnard_test(table, alpha, alternative)
        else:
            raise ValueError(f"不支持的检验方法: {method}")
        
        # 存储并返回结果
        self.results[method] = result
        return result
    
    def _z_test(self, table: np.ndarray, alpha: float, alternative: str) -> Dict:
        """两比例z检验"""
        n1, n2 = self.n_a, self.n_b
        p1, p2 = self.win_rate_a, self.win_rate_b
        
        # 计算合并比例
        p_pool = (self.wins_a + self.wins_b) / (n1 + n2)
        
        # 计算标准误
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        
        # 计算z统计量
        z = (p2 - p1) / se
        
        # 计算p值
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z)
        else:  # 'less'
            p_value = stats.norm.cdf(z)
        
        # 计算置信区间
        se_ci = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
        diff = p2 - p1
        
        if alternative == 'two-sided':
            z_crit = stats.norm.ppf(1 - alpha/2)
            ci_lower = diff - z_crit * se_ci
            ci_upper = diff + z_crit * se_ci
        elif alternative == 'greater':
            z_crit = stats.norm.ppf(1 - alpha)
            ci_lower = diff - z_crit * se_ci
            ci_upper = np.inf
        else:  # 'less'
            z_crit = stats.norm.ppf(1 - alpha)
            ci_lower = -np.inf
            ci_upper = diff + z_crit * se_ci
        
        return {
            'method': '两比例Z检验',
            'z_statistic': z,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': alpha,
            'alternative': alternative,
            'significant': p_value < alpha,
            'effect_size': diff,
            'recommendation': self._get_recommendation(p_value, alpha, diff)
        }
    
    def _chi2_test(self, table: np.ndarray, alpha: float) -> Dict:
        """卡方检验"""
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=True)
        
        # 计算效应量（Phi系数）
        phi = np.sqrt(chi2 / np.sum(table))
        
        return {
            'method': '卡方检验（带耶茨校正）',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'phi_coefficient': phi,
            'alpha': alpha,
            'significant': p_value < alpha,
            'recommendation': self._get_recommendation(p_value, alpha, phi)
        }
    
    def _fisher_test(self, table: np.ndarray, alpha: float, alternative: str) -> Dict:
        """Fisher精确检验"""
        oddsratio, p_value = stats.fisher_exact(table, alternative=alternative)
        
        return {
            'method': 'Fisher精确检验',
            'odds_ratio': oddsratio,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'significant': p_value < alpha,
            'recommendation': self._get_recommendation(p_value, alpha, oddsratio-1)
        }
    
    def _barnard_test(self, table: np.ndarray, alpha: float, alternative: str) -> Dict:
        """
        Barnard精确检验（蒙特卡洛近似）
        注：由于scipy没有内置Barnard检验，这里使用蒙特卡洛模拟
        """
        # 简化实现：使用置换检验代替
        n_permutations = 10000
        observed_diff = self.win_rate_b - self.win_rate_a
        
        # 合并数据
        total_wins = self.wins_a + self.wins_b
        total_games = self.n_a + self.n_b
        
        # 执行置换检验
        diffs = []
        for _ in range(n_permutations):
            # 随机分配胜场到两个策略
            perm_wins = np.random.hypergeometric(
                total_wins, total_games - total_wins, self.n_a + self.n_b, size=1
            )[0]
            perm_rate_a = perm_wins / self.n_a if self.n_a > 0 else 0
            perm_rate_b = (total_wins - perm_wins) / self.n_b if self.n_b > 0 else 0
            diffs.append(perm_rate_b - perm_rate_a)
        
        diffs = np.array(diffs)
        
        # 计算p值
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
        elif alternative == 'greater':
            p_value = np.mean(diffs >= observed_diff)
        else:  # 'less'
            p_value = np.mean(diffs <= observed_diff)
        
        return {
            'method': 'Barnard检验（蒙特卡洛近似）',
            'observed_diff': observed_diff,
            'p_value': p_value,
            'alpha': alpha,
            'alternative': alternative,
            'n_permutations': n_permutations,
            'significant': p_value < alpha,
            'recommendation': self._get_recommendation(p_value, alpha, observed_diff)
        }
    
    def _get_recommendation(self, p_value: float, alpha: float, effect: float) -> str:
        """根据检验结果给出推荐"""
        if p_value < alpha:
            if effect > 0:
                return f"推荐使用{self.name_b}（显著优于{self.name_a}）"
            else:
                return f"推荐使用{self.name_a}（显著优于{self.name_b}）"
        else:
            if abs(effect) > 0.05:  # 效应量大但统计不显著
                return f"差异不显著但效应量较大，建议收集更多数据"
            else:
                return f"无显著差异，两种策略效果相近"
    
    def get_power_analysis(self, alpha: float = 0.05, power: float = 0.8) -> Dict:
        """
        功效分析：计算检测到指定效应量所需样本量
        
        参数:
        ----------
        alpha : float
            显著性水平
        power : float
            期望的功效（1-β）
            
        返回:
        ----------
        Dict : 功效分析结果
        """
        from statsmodels.stats.power import NormalIndPower
        
        # 计算观测到的效应量（Cohen's h）
        h = 2 * np.arcsin(np.sqrt(self.win_rate_b)) - 2 * np.arcsin(np.sqrt(self.win_rate_a))
        
        # 计算当前样本量下的功效
        analysis = NormalIndPower()
        current_power = analysis.solve_power(
            effect_size=abs(h),
            nobs1=self.n_a,
            alpha=alpha,
            ratio=self.n_b/self.n_a if self.n_a > 0 else 1
        )
        
        # 计算达到指定功效所需样本量
        required_n = analysis.solve_power(
            effect_size=abs(h),
            alpha=alpha,
            power=power,
            ratio=self.n_b/self.n_a if self.n_a > 0 else 1
        )
        
        return {
            'observed_effect_size': abs(h),
            'current_power': current_power,
            'required_sample_size_per_group': int(np.ceil(required_n)),
            'current_total_samples': self.n_a + self.n_b,
            'required_total_samples': int(np.ceil(required_n * (1 + self.n_b/self.n_a))) if self.n_a > 0 else 0,
            'interpretation': self._interpret_power(current_power)
        }
    
    def _interpret_power(self, power: float) -> str:
        """解释功效结果"""
        if power < 0.5:
            return "功效很低，有很大可能漏掉真实差异"
        elif power < 0.8:
            return "功效不足，建议增加样本量"
        else:
            return "功效充足，检验结果可靠"


# 测试函数
def test_engine():
    """测试引擎功能"""
    print("测试A/B测试引擎...")
    print("=" * 50)
    
    # 创建引擎实例
    engine = ABTestEngine("主流策略", "创新策略")
    
    # 设置测试数据
    engine.set_data(
        n_a=1000,
        win_rate_a=0.52,
        n_b=50,
        win_rate_b=0.62
    )
    
    # 显示基本统计
    print("\n基本统计数据:")
    stats_df = engine.get_basic_stats()
    print(stats_df.to_string(index=False))
    
    # 执行Z检验
    print("\n执行Z检验...")
    result = engine.run_test(method='z_test', alpha=0.05, alternative='greater')
    print(f"Z统计量: {result['z_statistic']:.4f}")
    print(f"P值: {result['p_value']:.4f}")
    print(f"是否显著: {result['significant']}")
    print(f"推荐: {result['recommendation']}")
    
    # 功效分析
    print("\n功效分析:")
    power_result = engine.get_power_analysis()
    print(f"当前功效: {power_result['current_power']:.2%}")
    print(f"每组所需样本量: {power_result['required_sample_size_per_group']}")
    
    return engine


if __name__ == "__main__":
    test_engine()
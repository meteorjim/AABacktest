#!/usr/bin/env python3
"""
历史市场时间段分析脚本

分析不同ETF配置在不同市场环境（牛市、熊市、震荡市）下的表现
每个时间段至少1-2年，确保有足够时间观察再平衡效果
"""

import pandas as pd
import numpy as np
from portfolio_backtester import PortfolioBacktester
import sys
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def get_market_periods():
    """
    定义2014-2025年的重要市场时间段

    Returns:
        List[Dict]: 市场时间段列表，包含名称、类型、开始日期、结束日期
    """
    market_periods = [
        # 牛市期间
        {
            'name': '大牛市',
            'description': '2014-2015年大牛市',
            'type': '牛市',
            'start_date': '2014-01-01',
            'end_date': '2015-12-31',
            'duration_months': 24
        },
        {
            'name': '蓝筹牛市',
            'description': '2016-2017年蓝筹牛市',
            'type': '牛市',
            'start_date': '2016-01-01',
            'end_date': '2017-12-31',
            'duration_months': 24
        },
        {
            'name': '核心资产牛市',
            'description': '2019-2020年核心资产牛市',
            'type': '牛市',
            'start_date': '2019-01-01',
            'end_date': '2020-12-31',
            'duration_months': 24
        },
        {
            'name': 'AI结构性行情',
            'description': '2022-2023年AI结构性行情',
            'type': '牛市',
            'start_date': '2022-11-01',
            'end_date': '2023-12-31',
            'duration_months': 14
        },

        # 熊市期间
        {
            'name': '贸易战熊市',
            'description': '2018-2019年贸易战熊市',
            'type': '熊市',
            'start_date': '2018-01-01',
            'end_date': '2019-12-31',
            'duration_months': 24
        },
        {
            'name': '调整期',
            'description': '2021-2022年调整期',
            'type': '熊市',
            'start_date': '2021-02-01',
            'end_date': '2022-12-31',
            'duration_months': 23
        },
        {
            'name': '震荡市',
            'description': '2023-2024年震荡市',
            'type': '熊市',
            'start_date': '2023-01-01',
            'end_date': '2024-10-31',
            'duration_months': 22
        },

        # 特殊时期
        {
            'name': '完整周期',
            'description': '2016-2020年完整市场周期',
            'type': '完整周期',
            'start_date': '2016-01-01',
            'end_date': '2020-12-31',
            'duration_months': 60
        },
        {
            'name': '疫情特殊期',
            'description': '2020-2021年疫情特殊期',
            'type': '特殊期',
            'start_date': '2020-01-01',
            'end_date': '2021-12-31',
            'duration_months': 24
        }
    ]

    return market_periods


def get_weight_configurations():
    """
    获取测试的权重配置

    Returns:
        List[Tuple]: 权重配置列表，包含权重数组和配置名称
    """
    configurations = [
        # 原有包含国债的配置
        ([0.25, 0.25, 0.25, 0.25], "等权重配置"),
        ([0.40, 0.30, 0.15, 0.15], "偏债券配置"),
        ([0.20, 0.40, 0.20, 0.20], "偏红利配置"),
        ([0.15, 0.25, 0.10, 0.50], "偏纳斯达克配置"),
        ([0.10, 0.20, 0.60, 0.10], "偏黄金配置"),
        ([0.50, 0.20, 0.20, 0.10], "高债券配置"),
        ([0.05, 0.35, 0.10, 0.50], "高股票配置"),
        ([0.30, 0.35, 0.05, 0.30], "低黄金配置"),
        ([0.20, 0.30, 0.30, 0.20], "均衡偏商品配置"),
        ([0.35, 0.35, 0.15, 0.15], "保守股票配置"),
        ([0.10, 0.45, 0.05, 0.40], "激进配置"),
        ([0.60, 0.15, 0.15, 0.10], "超保守配置"),

        # 新增无国债配置
        # 纯资产配置
        ([0.00, 1.00, 0.00, 0.00], "纯红利配置(无国债)"),
        ([0.00, 0.00, 1.00, 0.00], "纯黄金配置(无国债)"),
        ([0.00, 0.00, 0.00, 1.00], "纯纳斯达克配置(无国债)"),

        # 双资产均衡配置
        ([0.00, 0.50, 0.50, 0.00], "红利+黄金均衡(无国债)"),
        ([0.00, 0.50, 0.00, 0.50], "红利+纳斯达克(无国债)"),
        ([0.00, 0.00, 0.50, 0.50], "黄金+纳斯达克(无国债)"),

        # 三资产均衡配置
        ([0.00, 0.40, 0.30, 0.30], "低红利均衡(无国债)"),
        ([0.00, 0.30, 0.40, 0.30], "低黄金均衡(无国债)"),
        ([0.00, 0.30, 0.30, 0.40], "低纳斯达克均衡(无国债)"),
        ([0.00, 0.33, 0.33, 0.34], "三资产均衡(无国债)"),

        # 红利主导配置
        ([0.00, 0.60, 0.20, 0.20], "红利主导(无国债)"),
        ([0.00, 0.70, 0.15, 0.15], "高红利配置(无国债)"),
        ([0.00, 0.80, 0.10, 0.10], "超高红利(无国债)"),

        # 黄金主导配置
        ([0.00, 0.20, 0.60, 0.20], "黄金主导(无国债)"),
        ([0.00, 0.15, 0.70, 0.15], "高黄金配置(无国债)"),
        ([0.00, 0.10, 0.80, 0.10], "超高黄金(无国债)"),

        # 纳斯达克主导配置
        ([0.00, 0.20, 0.20, 0.60], "纳斯达克主导(无国债)"),
        ([0.00, 0.15, 0.15, 0.70], "高纳斯达克配置(无国债)"),
        ([0.00, 0.10, 0.10, 0.80], "超高纳斯达克(无国债)"),

        # 激进成长配置
        ([0.00, 0.25, 0.25, 0.50], "激进成长(无国债)"),
        ([0.00, 0.20, 0.20, 0.60], "超级成长(无国债)"),
        ([0.00, 0.15, 0.15, 0.70], "极端成长(无国债)"),
    ]
    return configurations


def test_configuration_in_period(period, weights, config_name, etf_codes):
    """
    在指定市场时间段测试某个配置

    Args:
        period: 市场时间段字典
        weights: 权重数组
        config_name: 配置名称
        etf_codes: ETF代码列表

    Returns:
        Dict: 测试结果
    """
    try:
        # 创建回测实例
        backtester = PortfolioBacktester(
            etf_codes=etf_codes,
            weights=weights,
            enable_rebalancing=True,
            rebalance_freq='yearly',
            enable_dca=False,              # 不启用定投，便于观察纯粹配置效果
            start_date=period['start_date'],
            end_date=period['end_date'],
            initial_capital=100000,       # 初始资金10万元
            transaction_cost=0.001,       # 0.1%交易成本
            risk_free_rate=0.02,         # 无风险利率2%
            verbose_trading=False,        # 简化模式
            force_refresh=False           # 使用缓存
        )

        # 运行回测
        backtester.run_backtest()
        result_data = backtester.get_results()

        # 提取关键指标
        result = {
            '时间段名称': period['name'],
            '时间段描述': period['description'],
            '市场类型': period['type'],
            '开始日期': period['start_date'],
            '结束日期': period['end_date'],
            '持续月数': period['duration_months'],
            '权重配置': config_name,
            '国债权重': weights[0],
            '红利权重': weights[1],
            '黄金权重': weights[2],
            '纳斯达克权重': weights[3],
            '总收益率(%)': round(result_data['total_return'], 2),
            '年化收益率(%)': round(result_data['annual_return'], 2),
            '最大回撤(%)': round(result_data['max_drawdown'], 2),
            '回补时间(天)': result_data['max_drawdown_recovery_days'],
            '夏普比率': round(result_data['sharpe_ratio'], 3),
            '最终价值(元)': round(result_data['final_value'], 2),
            '总投入(元)': round(result_data['total_investment'], 2),
            '再平衡次数': result_data['rebalance_count'],
            '波动率(%)': round(result_data['volatility'], 2)
        }

        return result

    except Exception as e:
        print(f"测试失败 - {period['name']}, {config_name}: {str(e)}")
        # 返回错误结果
        return {
            '时间段名称': period['name'],
            '时间段描述': period['description'],
            '市场类型': period['type'],
            '权重配置': config_name,
            '总收益率(%)': 'ERROR',
            '年化收益率(%)': 'ERROR',
            '最大回撤(%)': 'ERROR',
            '夏普比率': 'ERROR',
            '错误信息': str(e)
        }


def run_market_period_analysis():
    """
    运行完整的市场时间段分析

    Returns:
        pd.DataFrame: 所有测试结果的DataFrame
    """
    print("=" * 100)
    print("历史市场时间段配置分析")
    print("=" * 100)

    # 获取市场时间段和配置
    market_periods = get_market_periods()
    configurations = get_weight_configurations()
    etf_codes = ['511010', '510880', '518880', '513100']  # 国债、红利、黄金、纳斯达克

    print(f"ETF组合: {etf_codes}")
    print(f"市场时间段数量: {len(market_periods)}")
    print(f"配置数量: {len(configurations)}")
    print(f"总测试数量: {len(market_periods) * len(configurations)}")
    print("-" * 100)

    # 存储所有结果
    all_results = []

    # 逐个时间段测试
    for period in market_periods:
        print(f"\n测试时间段: {period['name']} ({period['description']})")
        print(f"时间范围: {period['start_date']} 至 {period['end_date']} ({period['duration_months']}个月)")
        print("-" * 80)

        # 测试所有配置
        for weights, config_name in configurations:
            print(f"  测试配置: {config_name} ... ", end="")

            result = test_configuration_in_period(period, weights, config_name, etf_codes)
            all_results.append(result)

            if result['总收益率(%)'] != 'ERROR':
                print(f"完成 (收益率: {result['总收益率(%)']:.2f}%, 夏普: {result['夏普比率']:.3f})")
            else:
                print(f"失败 ({result.get('错误信息', '未知错误')})")

    # 创建DataFrame
    df = pd.DataFrame(all_results)

    # 保存结果
    csv_filename = 'market_periods_analysis.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 100)
    print("分析完成！")
    print(f"结果已保存到: {csv_filename}")

    # 统计信息
    successful_tests = df[df['总收益率(%)'] != 'ERROR']
    failed_tests = df[df['总收益率(%)'] == 'ERROR']

    print(f"成功测试: {len(successful_tests)}")
    print(f"失败测试: {len(failed_tests)}")

    return df


def analyze_period_performance(df):
    """
    分析各时间段的表现排名

    Args:
        df: 测试结果DataFrame
    """
    print("\n" + "=" * 80)
    print("各时间段表现排名分析")
    print("=" * 80)

    # 过滤成功的结果
    successful_df = df[df['总收益率(%)'] != 'ERROR'].copy()

    if successful_df.empty:
        print("没有成功的测试结果可供分析")
        return

    # 按时间段分组分析
    for period_name in successful_df['时间段名称'].unique():
        print(f"\n【{period_name}】表现排名:")
        period_data = successful_df[successful_df['时间段名称'] == period_name]

        # 按总收益率排序
        period_data_sorted = period_data.sort_values('总收益率(%)', ascending=False)

        print("收益率排名:")
        for i, (_, row) in enumerate(period_data_sorted.iterrows(), 1):
            print(f"  {i:2d}. {row['权重配置']:12s} - 收益率: {row['总收益率(%)']:7.2f}% "
                  f"(年化: {row['年化收益率(%)']:6.2f}%, 夏普: {row['夏普比率']:.3f})")

        # 按夏普比率排序
        sharpe_sorted = period_data.sort_values('夏普比率', ascending=False)
        print("\n夏普比率排名:")
        for i, (_, row) in enumerate(sharpe_sorted.iterrows(), 1):
            print(f"  {i:2d}. {row['权重配置']:12s} - 夏普: {row['夏普比率']:6.3f} "
                  f"(收益率: {row['总收益率(%)']:7.2f}%, 最大回撤: {row['最大回撤(%)']:6.2f}%)")

        print("-" * 60)


def analyze_market_type_performance(df):
    """
    分析不同市场类型下的配置表现

    Args:
        df: 测试结果DataFrame
    """
    print("\n" + "=" * 80)
    print("市场类型表现分析")
    print("=" * 80)

    # 过滤成功的结果
    successful_df = df[df['总收益率(%)'] != 'ERROR'].copy()

    if successful_df.empty:
        print("没有成功的测试结果可供分析")
        return

    # 按市场类型分组
    market_types = successful_df['市场类型'].unique()

    for market_type in market_types:
        print(f"\n【{market_type}】平均表现:")
        type_data = successful_df[successful_df['市场类型'] == market_type]

        # 计算各配置的平均表现
        config_performance = type_data.groupby('权重配置').agg({
            '总收益率(%)': 'mean',
            '年化收益率(%)': 'mean',
            '最大回撤(%)': 'mean',
            '夏普比率': 'mean',
            '波动率(%)': 'mean'
        }).round(2)

        # 按平均收益率排序
        config_performance_sorted = config_performance.sort_values('总收益率(%)', ascending=False)

        print("平均收益率排名:")
        for config_name, row in config_performance_sorted.iterrows():
            print(f"  {config_name:12s} - 收益率: {row['总收益率(%)']:7.2f}% "
                  f"(年化: {row['年化收益率(%)']:6.2f}%, 夏普: {row['夏普比率']:6.3f}, "
                  f"最大回撤: {row['最大回撤(%)']:6.2f}%)")

        print("-" * 60)


def create_performance_heatmap(df):
    """
    创建配置表现热力图

    Args:
        df: 测试结果DataFrame
    """
    try:
        # 过滤成功的结果
        successful_df = df[df['总收益率(%)'] != 'ERROR'].copy()

        if successful_df.empty:
            print("没有成功的数据可用于生成热力图")
            return

        # 创建收益率热力图数据
        pivot_returns = successful_df.pivot_table(
            index='权重配置',
            columns='时间段名称',
            values='总收益率(%)',
            fill_value=0
        )

        # 创建夏普比率热力图数据
        pivot_sharpe = successful_df.pivot_table(
            index='权重配置',
            columns='时间段名称',
            values='夏普比率',
            fill_value=0
        )

        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('总收益率(%) 热力图', '夏普比率 热力图'),
            vertical_spacing=0.15
        )

        # 添加收益率热力图
        fig.add_trace(
            go.Heatmap(
                z=pivot_returns.values,
                x=pivot_returns.columns,
                y=pivot_returns.index,
                colorscale='RdYlGn',
                text=pivot_returns.round(1).values,
                texttemplate='%{text}%',
                textfont=dict(size=10),
                hoverongaps=False,
                colorbar=dict(title="收益率(%)", x=1.02),
                hovertemplate='配置: %{y}<br>时间段: %{x}<br>收益率: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 添加夏普比率热力图
        fig.add_trace(
            go.Heatmap(
                z=pivot_sharpe.values,
                x=pivot_sharpe.columns,
                y=pivot_sharpe.index,
                colorscale='RdYlGn',
                text=pivot_sharpe.round(2).values,
                texttemplate='%{text}',
                textfont=dict(size=10),
                hoverongaps=False,
                colorbar=dict(title="夏普比率", x=1.02),
                hovertemplate='配置: %{y}<br>时间段: %{x}<br>夏普比率: %{z:.3f}<extra></extra>'
            ),
            row=2, col=1
        )

        # 更新布局
        fig.update_layout(
            title={
                'text': '配置在不同市场时间段下的表现热力图',
                'x': 0.5,
                'font': {'size': 20}
            },
            template='plotly_white',
            height=1000,
            width=1200,
            showlegend=False
        )

        # 更新坐标轴
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=1)

        # 显示图表
        fig.show()

        print("\n热力图已生成，显示了各配置在不同时间段的表现对比")

    except Exception as e:
        print(f"生成热力图时出错: {str(e)}")


def print_summary_statistics(df):
    """
    打印汇总统计信息

    Args:
        df: 测试结果DataFrame
    """
    print("\n" + "=" * 80)
    print("汇总统计信息")
    print("=" * 80)

    # 过滤成功的结果
    successful_df = df[df['总收益率(%)'] != 'ERROR'].copy()

    if successful_df.empty:
        print("没有成功的测试结果可供分析")
        return

    # 各配置整体表现
    print("\n各配置整体表现排名:")
    config_overall = successful_df.groupby('权重配置').agg({
        '总收益率(%)': ['mean', 'std'],
        '年化收益率(%)': 'mean',
        '最大回撤(%)': 'mean',
        '夏普比率': 'mean',
        '波动率(%)': 'mean'
    }).round(2)

    # 计算综合评分（收益率和夏普比率的加权平均）
    config_overall['综合评分'] = (
        config_overall[('总收益率(%)', 'mean')] * 0.5 +
        config_overall[('夏普比率', 'mean')] * 100 * 0.5
    ).round(2)

    # 按综合评分排序
    config_overall_sorted = config_overall.sort_values('综合评分', ascending=False)

    for i, (config_name, row) in enumerate(config_overall_sorted.iterrows(), 1):
        mean_return = float(row[('总收益率(%)', 'mean')])
        std_return = float(row[('总收益率(%)', 'std')])
        sharpe = float(row[('夏普比率', 'mean')])
        max_dd = float(row[('最大回撤(%)', 'mean')])
        score = float(row['综合评分'])

        print(f"  {i:2d}. {config_name:12s} - 综合评分: {score:6.2f} | "
              f"平均收益: {mean_return:6.2f}%±{std_return:5.2f}% | "
              f"夏普: {sharpe:5.3f} | 最大回撤: {max_dd:6.2f}%")

    # 找出最佳配置
    best_return_config = config_overall_sorted.index[0]
    best_sharpe_config = config_overall_sorted.sort_values(('夏普比率', 'mean'), ascending=False).index[0]
    most_stable_config = config_overall_sorted.sort_values(('总收益率(%)', 'std')).index[0]

    print(f"\n🏆 最佳综合配置: {best_return_config}")
    print(f"🎯 最佳夏普配置: {best_sharpe_config}")
    print(f"🛡️  最稳定配置: {most_stable_config}")


if __name__ == "__main__":
    try:
        # 运行分析
        results_df = run_market_period_analysis()

        if not results_df.empty:
            # 各时间段表现排名
            analyze_period_performance(results_df)

            # 市场类型表现分析
            analyze_market_type_performance(results_df)

            # 汇总统计
            print_summary_statistics(results_df)

            # 生成热力图
            create_performance_heatmap(results_df)

            print(f"\n✅ 分析完成！共分析了 {len(results_df)} 个测试结果")
            print("📊 详细结果已保存到 market_periods_analysis.csv")

    except KeyboardInterrupt:
        print("\n❌ 分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {str(e)}")
        sys.exit(1)
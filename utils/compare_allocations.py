#!/usr/bin/env python3
"""
资产配置与阈值再平衡对比分析

测试不同的资产配置比例和再平衡阈值组合，
对比年化收益、最大回撤等指标差异。
"""

from portfolio_backtester import PortfolioBacktester
import pandas as pd
from datetime import datetime


# 基础资产代码 (指数数据)
ASSET_CODES = ['000012', '000015@0.03', 'XAUUSD:CUR', 'QQQ']  # 国债、红利(3%分红)、黄金、纳指
ASSET_NAMES = ['国债', '红利', '黄金', '纳指']


# 资产配置方案 (国债:红利:黄金:纳指)
ALLOCATION_SCHEMES = {
    "保守型": [5, 2, 1, 1],      # 5:2:1:1
    "稳健型": [3, 3, 2, 2],      # 3:3:2:2
    "平衡型": [2, 2, 2, 2],      # 2:2:2:2
    "进取型": [1, 2, 2, 3],      # 1:2:2:3
    "激进型": [0.5, 1, 2, 4],    # 0.5:1:2:4
    "股金型": [0, 1, 2, 5],      # 0:1:2:5
    "红利增强": [3, 4, 1, 1],     # 3:4:1:1
    "科技成长": [1, 1, 1, 5],     # 1:1:1:5
}


# 再平衡阈值方案
THRESHOLDS = [0, 0.01, 0.03, 0.05]  # 无阈值, 1%, 3%, 5%


def run_single_backtest(scheme_name: str, weights: list, threshold: float) -> dict:
    """运行单个回测并返回关键指标"""
    print(f"  测试: {scheme_name} | 阈值: {threshold*100:.0f}% ...", end=" ")

    backtester = PortfolioBacktester(
        etf_codes=ASSET_CODES,
        weights=weights,
        start_date='2005-01-01',
        end_date='2026-02-28',
        initial_capital=100000,
        enable_rebalancing=True,
        rebalance_freq='yearly',
        rebalance_threshold=threshold,
        enable_dca=True,
        dca_amount=100000,
        dca_freq='yearly',
        transaction_cost=0,
        risk_free_rate=0.02,
        force_refresh=False,
        verbose_trading=False,
        show_daily_logs=False,
        save_html=False,
    )

    backtester.run_backtest()

    print("完成")
    r = backtester.results
    # r中的值已经是百分比形式，转换为小数
    return {
        "cagr": r.get("annual_return", 0) / 100,
        "max_drawdown": abs(r.get("max_drawdown", 0)) / 100,
        "sharpe": r.get("sharpe_ratio", 0),
        "total_return": r.get("total_return", 0),
        "volatility": r.get("volatility", 0),
        "rebalance_count": len(backtester.rebalance_dates),
    }


def run_comparison():
    """运行完整对比测试"""
    print("=" * 80)
    print("资产配置与阈值再平衡对比分析 (指数数据)")
    print("=" * 80)
    print(f"测试期间: 2005-01-01 至 2026-02-28 (21年)")
    print(f"初始资金: 100,000 元 | 年度定投: 100,000 元")
    print(f"配置方案: {len(ALLOCATION_SCHEMES)} 套")
    print(f"阈值方案: {len(THRESHOLDS)} 种 (无阈值, 1%, 3%, 5%)")
    print(f"总测试数: {len(ALLOCATION_SCHEMES) * len(THRESHOLDS)}")
    print("=" * 80)
    print()

    # 存储结果
    all_results = {}

    for scheme_name, weights in ALLOCATION_SCHEMES.items():
        print(f"【{scheme_name}】权重: {weights}")
        all_results[scheme_name] = {}
        for threshold in THRESHOLDS:
            result = run_single_backtest(scheme_name, weights, threshold)
            all_results[scheme_name][threshold] = result
        print()

    return all_results


def format_ratio(weights: list) -> str:
    """格式化权重比例为字符串"""
    # 将权重转换为整数比例
    min_w = min(w for w in weights if w > 0)
    int_weights = [int(round(w / min_w)) for w in weights]
    return f"{' '.join(map(str, int_weights))}".replace(' ', ':')


def create_summary_table(results: dict):
    """创建汇总表格"""
    print("\n" + "=" * 100)
    print("年化收益率对比 (%) | 资产比例: 国债:红利:黄金:纳指")
    print("=" * 100)

    cagr_data = []
    for scheme_name, weights in ALLOCATION_SCHEMES.items():
        row = {"配置方案": scheme_name, "比例": format_ratio(weights)}
        for threshold in THRESHOLDS:
            val = results[scheme_name][threshold]["cagr"] * 100
            row[f"{threshold*100:.0f}%"] = f"{val:.2f}"
        cagr_data.append(row)

    df_cagr = pd.DataFrame(cagr_data)
    df_cagr = df_cagr.set_index(["配置方案", "比例"])
    print(df_cagr.to_string())

    print("\n" + "=" * 100)
    print("最大回撤对比 (%) | 资产比例: 国债:红利:黄金:纳指")
    print("=" * 100)

    dd_data = []
    for scheme_name, weights in ALLOCATION_SCHEMES.items():
        row = {"配置方案": scheme_name, "比例": format_ratio(weights)}
        for threshold in THRESHOLDS:
            val = results[scheme_name][threshold]["max_drawdown"] * 100
            row[f"{threshold*100:.0f}%"] = f"{val:.2f}"
        dd_data.append(row)

    df_dd = pd.DataFrame(dd_data)
    df_dd = df_dd.set_index(["配置方案", "比例"])
    print(df_dd.to_string())

    print("\n" + "=" * 100)
    print("夏普比率对比 | 资产比例: 国债:红利:黄金:纳指")
    print("=" * 100)

    sharpe_data = []
    for scheme_name, weights in ALLOCATION_SCHEMES.items():
        row = {"配置方案": scheme_name, "比例": format_ratio(weights)}
        for threshold in THRESHOLDS:
            val = results[scheme_name][threshold]["sharpe"]
            row[f"{threshold*100:.0f}%"] = f"{val:.3f}"
        sharpe_data.append(row)

    df_sharpe = pd.DataFrame(sharpe_data)
    df_sharpe = df_sharpe.set_index(["配置方案", "比例"])
    print(df_sharpe.to_string())

    print("\n" + "=" * 100)
    print("关键发现摘要")
    print("=" * 100)

    # 找出各指标最优组合
    best_cagr = max(
        (s, t, results[s][t]["cagr"])
        for s in ALLOCATION_SCHEMES for t in THRESHOLDS
    )
    best_dd = min(
        (s, t, results[s][t]["max_drawdown"])
        for s in ALLOCATION_SCHEMES for t in THRESHOLDS
    )
    best_sharpe = max(
        (s, t, results[s][t]["sharpe"])
        for s in ALLOCATION_SCHEMES for t in THRESHOLDS
    )

    print(f"最高年化收益: {best_cagr[0]} (比例{format_ratio(ALLOCATION_SCHEMES[best_cagr[0]])}) + {best_cagr[1]*100:.0f}%阈值 = {best_cagr[2]*100:.2f}%")
    print(f"最小最大回撤: {best_dd[0]} (比例{format_ratio(ALLOCATION_SCHEMES[best_dd[0]])}) + {best_dd[1]*100:.0f}%阈值 = {best_dd[2]*100:.2f}%")
    print(f"最高夏普比率: {best_sharpe[0]} (比例{format_ratio(ALLOCATION_SCHEMES[best_sharpe[0]])}) + {best_sharpe[1]*100:.0f}%阈值 = {best_sharpe[2]:.3f}")

    print("\n" + "=" * 100)


def save_detailed_results(results: dict):
    """保存详细结果到CSV"""
    detailed_data = []

    for scheme_name, weights in ALLOCATION_SCHEMES.items():
        for threshold in THRESHOLDS:
            r = results[scheme_name][threshold]
            detailed_data.append({
                "配置方案": scheme_name,
                "比例(国债:红利:黄金:纳指)": format_ratio(weights),
                "再平衡阈值": f"{threshold*100:.0f}%",
                "年化收益率": f"{r['cagr']*100:.2f}%",
                "最大回撤": f"{r['max_drawdown']*100:.2f}%",
                "夏普比率": f"{r['sharpe']:.3f}",
                "总收益率": f"{r['total_return']:.2f}%",
                "再平衡次数": r['rebalance_count'],
            })

    df = pd.DataFrame(detailed_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"allocation_comparison_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至: {filename}")


if __name__ == "__main__":
    results = run_comparison()
    create_summary_table(results)
    save_detailed_results(results)

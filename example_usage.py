#!/usr/bin/env python3
"""
资产配置回测系统使用示例

这个脚本展示了如何使用PortfolioBacktester进行不同配置的资产回测
"""

from portfolio_backtester import PortfolioBacktester



if __name__ == "__main__":

    print("=" * 50)
    print("基础资产配置示例")
    print("=" * 50)

                  # 国债.     红利.      黄金.     纳斯达克.   创业板
    # etf_codes=['511010', '510880', '518880', '513100', '159915'],
    backtester = PortfolioBacktester(
                    # 国债.   红利.      黄金.     纳斯达克.
        etf_codes=['511010', '510880', '518880', '513100'],
        weights=[0.2, 0.4, 0.2, 0.2],
        # etf_codes=['159915'],
        # weights=[1],
        enable_rebalancing=True,
        rebalance_freq='yearly',
        start_date='2016-01-01',
        end_date='2025-10-01',
        transaction_cost=0,
        initial_capital=100000,
        force_refresh=False,      # 不强制刷新缓存
        enable_dca = True,
        dca_amount = 100000,
        dca_freq = 'yearly'
    )

    # 运行回测
    backtester.run_backtest()

    # 生成完整报告
    # backtester.generate_report(show_plot=True, show_benchmark=True)
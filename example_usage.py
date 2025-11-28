#!/usr/bin/env python3
"""
资产配置回测系统使用示例

这个脚本展示了如何使用PortfolioBacktester进行不同配置的资产回测
包括新的 verbose_trading 参数的使用
"""

from portfolio_backtester import PortfolioBacktester



if __name__ == "__main__":

    backtester = PortfolioBacktester(
                    # 国债.     红利.     黄金.     纳斯达克.
        etf_codes=['511010', '510880', '518880', '513100'],
        # weights=  [0.25,       0.25,      0.25,      0.25],
        weights=  [0.28,       0.28,      0.14,      0.30],
        # etf_codes=['601288', '601398', '601939', '601988'],
        # weights=  [0.25,       0.25,      0.25,      0.25],
        # etf_codes=['518880', '513100'],
        # weights=  [0.3,       0.7,      ],
        # etf_codes=['159915'],
        # weights=[1],
        enable_rebalancing=True,
        rebalance_freq='yearly',
        start_date='2025-02-25',
        end_date='2025-11-28',
        transaction_cost=0,
        initial_capital=600000,
        risk_free_rate=0.02,      # 设置无风险利率为2%
        force_refresh=False,      # 不强制刷新缓存
        verbose_trading=False,    # 简化模式，只显示基本交易信息
        # save_html=True,
        # enable_dca = True,
        # dca_amount = 100000,
        # dca_freq = 'yearly',
    )

    # 运行回测
    backtester.run_backtest()

    # 生成完整报告
    backtester.generate_report(show_plot=False)

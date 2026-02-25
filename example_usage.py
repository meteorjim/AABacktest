#!/usr/bin/env python3
"""
资产配置回测系统使用示例

这个脚本展示了如何使用PortfolioBacktester进行不同配置的资产回测
包括新的 verbose_trading 参数的使用
"""

from portfolio_backtester import PortfolioBacktester



if __name__ == "__main__":

    backtester = PortfolioBacktester(
        # --- 核心资产配置 ---
        #           国债.     红利(模拟3%分红).  黄金.     纳斯达克.
        etf_codes=['000012', '000015@0.03', 'XAUUSD:CUR', 'QQQ'],   # ETF代码列表：输入你要回测的ETF/股票代码
        weights=  [1,       1,      1,      1],         # 目标权重：对应上述代码的配置比例(会自动归一化)
        # etf_codes=['511010', '510880', '518880', '513100'],   # ETF代码列表：输入你要回测的ETF/股票代码
        # weights=  [1,       1,      1,      1],         # 目标权重：对应上述代码的配置比例(会自动归一化)
 
        # --- 再平衡策略 ---
        enable_rebalancing=True,          # 是否启用自动再平衡：True=开启, False=关闭
        rebalance_freq='yearly',          # 再平衡频率：'monthly'(月度), 'quarterly'(季度), 'yearly'(年度)
        # rebalance_threshold=0.03,         # 阈值再平衡：当任一资产权重偏离目标超过此值(如0.01=1%)时，强制触发再平衡

        # --- 回测时间与资金 ---
        start_date='2005-01-01',          # 回测开始日期 'YYYY-MM-DD'
        end_date='2026-02-01',            # 回测结束日期 'YYYY-MM-DD'
        initial_capital=100000,           # 初始投入资金(元)
        transaction_cost=0,               # 交易费率：单边买/卖手续费率 (如 0.0001 = 万分之一)
        risk_free_rate=0.02,              # 无风险利率：用于计算夏普比率 (0.02 = 2%)

        # --- 定投策略 (DCA) ---
        enable_dca = True,                # 是否开启定投：True=开启, False=关闭
        dca_amount = 100000,              # 每次定投金额(元)
        dca_freq = 'yearly',              # 定投频率：'monthly'(月度), 'yearly'(年度)

        # --- 系统显示设置 ---
        force_refresh=False,              # 是否强制刷新：True=重新下载数据, False=优先使用本地缓存
        verbose_trading=False,            # 详细交易日志：True=显示每笔具体买卖明细, False=只显示汇总
        show_daily_logs=False,            # 每日日志开关：True=显示每日定投/再平衡日志, False=静默模式(只看结果)
        save_html=True,                   # 结果保存：True=生成交互式HTML图表报告
    )

    # 运行回测
    backtester.run_backtest()

    # 生成完整报告
    backtester.generate_report(show_plot=True)

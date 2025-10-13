"""
步骤1：基础回测功能测试
使用fixtures提供的数据，但采用完全独立的计算逻辑验证PortfolioBacktester的正确性
"""

import pytest
import pandas as pd
import numpy as np
from portfolio_backtester import PortfolioBacktester


class TestBasicBacktest:
    """基础回测测试类"""

    def test_single_etf_initial_buy(self, single_etf_config, etf_data_single):
        """测试单只ETF初始建仓"""
        # 使用fixtures提供的独立数据
        etf_code = single_etf_config['etf_codes'][0]
        data = etf_data_single[etf_code]

        # 独立计算预期结果
        initial_capital = single_etf_config['initial_capital']
        open_price = data.iloc[0]['open']  # 第一天开盘价
        expected_shares = initial_capital / open_price  # 允许碎股
        expected_avg_cost = open_price

        # 运行PortfolioBacktester
        backtester = PortfolioBacktester(**single_etf_config)
        backtester.run_backtest()

        # 获取实际结果
        position = backtester.positions[etf_code]
        actual_shares = position['shares']
        actual_avg_cost = position['avg_cost']

        # 使用assert验证结果
        assert abs(actual_shares - expected_shares) < 0.001, \
            f"持仓股数不匹配: 期望 {expected_shares:.4f}, 实际 {actual_shares:.4f}"

        assert abs(actual_avg_cost - expected_avg_cost) < 0.001, \
            f"平均成本不匹配: 期望 {expected_avg_cost:.3f}, 实际 {actual_avg_cost:.3f}"

        # 验证现金余额接近0（因为允许碎股）
        assert backtester.cash < 1.0, \
            f"现金余额过多: {backtester.cash:.2f}"

    def test_single_etf_final_value(self, single_etf_config, etf_data_single):
        """测试单只ETF最终价值计算"""
        # 使用fixtures数据
        etf_code = single_etf_config['etf_codes'][0]
        data = etf_data_single[etf_code]

        # 运行回测
        backtester = PortfolioBacktester(**single_etf_config)
        backtester.run_backtest()

        # 独立计算最终价值
        position = backtester.positions[etf_code]
        last_close_price = data.iloc[-1]['close']  # 最后一天收盘价
        expected_final_value = position['shares'] * last_close_price + backtester.cash
        actual_final_value = backtester.get_results()['final_value']

        # 验证最终价值
        assert abs(actual_final_value - expected_final_value) < 1.0, \
            f"最终价值不匹配: 期望 {expected_final_value:.2f}, 实际 {actual_final_value:.2f}"

    def test_single_etf_total_return(self, single_etf_config):
        """测试单只ETF总收益率"""
        # 运行回测
        backtester = PortfolioBacktester(**single_etf_config)
        backtester.run_backtest()

        # 独立计算收益率
        results = backtester.get_results()
        initial_capital = single_etf_config['initial_capital']
        final_value = results['final_value']

        expected_return = (final_value / initial_capital - 1) * 100
        actual_return = results['total_return']

        # 验证收益率
        assert abs(actual_return - expected_return) < 0.01, \
            f"总收益率不匹配: 期望 {expected_return:.2f}%, 实际 {actual_return:.2f}%"

    def test_single_etf_daily_values(self, single_etf_config, etf_data_single):
        """测试单只ETF每日价值计算"""
        # 使用fixtures数据
        etf_code = single_etf_config['etf_codes'][0]
        data = etf_data_single[etf_code]

        # 运行回测
        backtester = PortfolioBacktester(**single_etf_config)
        backtester.run_backtest()

        # 验证前10天的每日价值
        position = backtester.positions[etf_code]
        shares = position['shares']

        for i in range(min(10, len(backtester.daily_values))):
            date = backtester.daily_dates[i]

            # 在独立数据中找到对应日期
            if date in data.index:
                close_price = data.loc[date, 'close']
                expected_value = shares * close_price
                actual_value = backtester.daily_values[i]

                # 允许小幅误差（现金余额影响）
                assert abs(actual_value - expected_value) < 10.0, \
                    f"日期 {date.strftime('%Y-%m-%d')} 价值计算错误: 期望 {expected_value:.2f}, 实际 {actual_value:.2f}"

    def test_multiple_etf_initial_buy(self, multiple_etf_config, etf_data_multiple):
        """测试多只ETF初始建仓"""
        # 使用fixtures数据
        etf_codes = multiple_etf_config['etf_codes']
        weights = multiple_etf_config['weights']
        initial_capital = multiple_etf_config['initial_capital']
        transaction_cost = multiple_etf_config['transaction_cost']

        # 运行回测
        backtester = PortfolioBacktester(**multiple_etf_config)
        backtester.run_backtest()

        # 独立验证每只ETF的建仓
        total_expected_cost = 0
        for i, etf_code in enumerate(etf_codes):
            if etf_code not in backtester.positions or etf_code not in etf_data_multiple:
                continue

            position = backtester.positions[etf_code]
            data = etf_data_multiple[etf_code]

            # 独立计算预期
            target_value = initial_capital * weights[i]
            open_price = data.iloc[0]['open']
            # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
            expected_shares = target_value / (open_price * (1 + transaction_cost))
            expected_cost = expected_shares * open_price * (1 + transaction_cost)

            total_expected_cost += expected_cost

            # 验证
            assert abs(position['shares'] - expected_shares) < 0.001, \
                f"{etf_code} 持仓股数错误: 期望 {expected_shares:.4f}, 实际 {position['shares']:.4f}"

        # 验证总成本合理
        assert total_expected_cost <= initial_capital, \
            f"总成本超过初始资金: {total_expected_cost:.2f} > {initial_capital:.2f}"

    def test_multiple_etf_portfolio_weights(self, multiple_etf_config, etf_data_multiple):
        """测试多只ETF组合权重分配"""
        # 使用fixtures数据
        etf_codes = multiple_etf_config['etf_codes']
        weights = multiple_etf_config['weights']

        # 运行回测
        backtester = PortfolioBacktester(**multiple_etf_config)
        backtester.run_backtest()

        # 验证最终持仓权重
        final_value = backtester.get_results()['final_value']

        for i, etf_code in enumerate(etf_codes):
            if etf_code not in backtester.positions or etf_code not in etf_data_multiple:
                continue

            position = backtester.positions[etf_code]
            data = etf_data_multiple[etf_code]
            last_close_price = data.iloc[-1]['close']

            market_value = position['shares'] * last_close_price
            actual_weight = market_value / final_value
            expected_weight = weights[i]

            # 允许一定误差（交易成本和剩余现金影响）
            assert abs(actual_weight - expected_weight) < 0.05, \
                f"{etf_code} 权重偏差过大: 期望 {expected_weight:.2f}, 实际 {actual_weight:.2f}"

    def test_performance_metrics_calculation(self, single_etf_config, etf_data_single):
        """测试性能指标计算"""
        # 运行回测
        backtester = PortfolioBacktester(**single_etf_config)
        backtester.run_backtest()

        results = backtester.get_results()

        # 独立计算最大回撤
        values = pd.Series(backtester.daily_values)
        rolling_max = values.expanding().max()
        drawdown = (values - rolling_max) / rolling_max
        expected_max_dd = drawdown.min() * 100
        actual_max_dd = results['max_drawdown']

        assert abs(actual_max_dd - expected_max_dd) < 0.1, \
            f"最大回撤计算错误: 期望 {expected_max_dd:.2f}%, 实际 {actual_max_dd:.2f}%"

        # 独立计算年化收益率（使用实际交易日）
        initial_capital = single_etf_config['initial_capital']
        final_value = results['final_value']

        # 使用实际交易日计算年化收益率
        trading_days = len(backtester.daily_values)
        years = trading_days / 252.0  # 假设一年252个交易日

        if years > 0:
            expected_annual_return = (final_value / initial_capital) ** (1/years) - 1
            expected_annual_return_pct = expected_annual_return * 100
            actual_annual_return = results['annual_return']

            # 调整容差，因为年化率计算可能有细微差异
            assert abs(actual_annual_return - expected_annual_return_pct) < 0.5, \
                f"年化收益率计算错误: 期望 {expected_annual_return_pct:.2f}%, 实际 {actual_annual_return:.2f}%"

    def test_transaction_cost_impact(self, etf_data_single):
        """测试交易成本影响"""
        config = {
            'etf_codes': ['511010'],
            'weights': [1.0],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'initial_capital': 100000,
            'enable_dca': False
        }

        # 无交易成本
        backtester_no_cost = PortfolioBacktester(**config, transaction_cost=0)
        backtester_no_cost.run_backtest()

        # 有交易成本
        backtester_with_cost = PortfolioBacktester(**config, transaction_cost=0.001)
        backtester_with_cost.run_backtest()

        # 验证交易成本影响
        final_no_cost = backtester_no_cost.get_results()['final_value']
        final_with_cost = backtester_with_cost.get_results()['final_value']

        assert final_with_cost < final_no_cost, \
            "有交易成本的最终价值应该更低"

        # 验证成本差异合理
        cost_ratio = (final_no_cost - final_with_cost) / final_no_cost
        assert 0 < cost_ratio < 0.002, \
            f"交易成本比例不合理: {cost_ratio:.4f}"

    @pytest.mark.parametrize("initial_capital", [10000, 50000, 100000])
    def test_different_capital_amounts(self, initial_capital, etf_data_single):
        """测试不同初始资金量"""
        config = {
            'etf_codes': ['511010'],
            'weights': [1.0],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'initial_capital': initial_capital,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证最终价值合理
        final_value = backtester.get_results()['final_value']
        assert final_value > 0, \
            f"资金量 {initial_capital} 的最终价值应大于0"

        # 验证现金利用率
        cash_ratio = backtester.cash / initial_capital
        assert cash_ratio < 0.01, \
            f"资金量 {initial_capital} 的现金利用率过低: {1-cash_ratio:.4f}"

    def test_data_integrity(self, etf_data_single, etf_data_multiple):
        """测试数据完整性"""
        # 验证单只ETF数据
        assert len(etf_data_single) > 0, "单只ETF数据不应为空"
        etf_code = list(etf_data_single.keys())[0]
        data = etf_data_single[etf_code]

        # 验证必要列存在
        required_columns = ['open', 'close', 'high', 'low', 'volume']
        for col in required_columns:
            assert col in data.columns, f"数据缺少列: {col}"

        # 验证数据量合理
        assert len(data) > 20, f"{etf_code} 数据量过少: {len(data)}"

        # 验证多只ETF数据
        assert len(etf_data_multiple) >= 2, "多只ETF数据不应为空"
        for etf_code, data in etf_data_multiple.items():
            assert len(data) > 20, f"{etf_code} 数据量过少: {len(data)}"
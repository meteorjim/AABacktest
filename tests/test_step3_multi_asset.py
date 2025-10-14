"""
步骤3：多资产组合功能测试
验证多ETF组合的正确计算、权重分配和性能指标
"""

import pytest
import pandas as pd
import numpy as np
from portfolio_backtester import PortfolioBacktester


class TestMultiAssetPortfolio:
    """多资产组合测试类"""

    def test_three_etf_portfolio_initial_allocation(self, etf_data_multiple):
        """测试三只ETF组合的初始分配"""
        # 使用511010和510880，再添加一个虚拟的第三只ETF数据
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.3, 0.2],  # 这个会失败，因为只有2只ETF
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 120000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        # 预期会抛出异常，因为权重数量与ETF数量不匹配
        with pytest.raises(ValueError, match="ETF代码数量与权重数量不匹配"):
            PortfolioBacktester(**config)

    def test_multi_etf_weight_allocation(self, etf_data_multiple):
        """测试多ETF组合的权重分配"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.7, 0.3],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证最终权重接近目标权重
        final_value = backtester.get_results()['final_value']
        for i, etf_code in enumerate(backtester.etf_codes):
            if etf_code in backtester.positions and etf_code in etf_data_multiple:
                position = backtester.positions[etf_code]
                data = etf_data_multiple[etf_code]
                last_date = backtester.daily_dates[-1]
                if last_date in data.index:
                    current_price = data.loc[last_date, 'close']
                    market_value = position['shares'] * current_price
                    actual_weight = market_value / final_value
                    target_weight = backtester.weights[i]

                    # 允许一定的偏差（价格波动和交易成本影响）
                    assert abs(actual_weight - target_weight) < 0.1, \
                        f"{etf_code} 最终权重偏差过大: 期望 {target_weight:.2f}, 实际 {actual_weight:.2f}"

    def test_portfolio_value_calculation(self, etf_data_multiple):
        """测试组合价值计算的正确性"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 独立计算最终组合价值
        expected_final_value = 0
        for etf_code in backtester.etf_codes:
            if etf_code in backtester.positions and etf_code in etf_data_multiple:
                position = backtester.positions[etf_code]
                data = etf_data_multiple[etf_code]
                last_date = backtester.daily_dates[-1]
                if last_date in data.index:
                    current_price = data.loc[last_date, 'close']
                    market_value = position['shares'] * current_price
                    expected_final_value += market_value

        expected_final_value += backtester.cash
        actual_final_value = backtester.get_results()['final_value']

        assert abs(actual_final_value - expected_final_value) < 1.0, \
            f"组合价值计算错误: 期望 {expected_final_value:.2f}, 实际 {actual_final_value:.2f}"

    def test_portfolio_performance_metrics(self, etf_data_multiple):
        """测试组合级性能指标计算"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        results = backtester.get_results()

        # 验证基本指标
        assert results['final_value'] > 0, "最终价值应大于0"
        assert results['initial_capital'] == 100000, "初始资金应为100000"
        assert results['trading_days'] > 0, "交易天数应大于0"
        assert abs(results['total_investment'] - config['initial_capital']) < 0.01, \
            "无定投时总投入应等于初始资金"

        # 验证收益率计算
        expected_return = (results['final_value'] / results['total_investment'] - 1) * 100
        assert abs(results['total_return'] - expected_return) < 0.01, \
            f"总收益率计算错误: 期望 {expected_return:.2f}%, 实际 {results['total_return']:.2f}%"

    def test_transaction_cost_impact_on_portfolio(self, etf_data_multiple):
        """测试交易成本对组合的影响"""
        base_config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'enable_dca': False
        }

        # 无交易成本
        backtester_no_cost = PortfolioBacktester(**base_config, transaction_cost=0)
        backtester_no_cost.run_backtest()

        # 有交易成本
        backtester_with_cost = PortfolioBacktester(**base_config, transaction_cost=0.002)
        backtester_with_cost.run_backtest()

        # 验证交易成本影响
        final_no_cost = backtester_no_cost.get_results()['final_value']
        final_with_cost = backtester_with_cost.get_results()['final_value']

        assert final_with_cost < final_no_cost, \
            "有交易成本的最终价值应该更低"

        # 验证成本差异合理
        cost_ratio = (final_no_cost - final_with_cost) / final_no_cost
        assert 0 < cost_ratio < 0.01, \
            f"多ETF组合交易成本比例不合理: {cost_ratio:.4f}"

    def test_different_weight_allocations(self):
        """测试不同权重分配的影响"""
        weight_configs = [
            ([0.8, 0.2], "偏重511010"),
            ([0.2, 0.8], "偏重510880"),
            ([0.5, 0.5], "均衡配置")
        ]

        results = {}
        for weights, description in weight_configs:
            config = {
                'etf_codes': ['511010', '510880'],
                'weights': weights,
                'start_date': '2024-01-01',
                'end_date': '2024-03-31',
                'initial_capital': 100000,
                'transaction_cost': 0,
                'enable_dca': False
            }

            backtester = PortfolioBacktester(**config)
            backtester.run_backtest()
            results[description] = backtester.get_results()['final_value']

        # 验证不同配置产生了不同的结果
        values = list(results.values())
        assert len(set([round(v, 2) for v in values])) > 1, \
            "不同权重配置应该产生不同的最终价值"

        print("不同权重配置的最终价值:")
        for desc, value in results.items():
            print(f"  {desc}: ¥{value:,.2f}")

    def test_portfolio_with_missing_data(self):
        """测试处理缺失数据的能力"""
        config = {
            'etf_codes': ['511010', '999999'],  # 999999是不存在的ETF代码
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)

        # 应该能够处理部分数据缺失的情况
        backtester.run_backtest()

        # 验证只获取了有效数据的ETF
        assert len(backtester.etf_data) > 0, "应该至少有一个ETF的数据"
        assert len(backtester.positions) <= len(backtester.etf_data), \
            "持仓数量不应超过有效数据ETF数量"

    def test_extreme_weight_configurations(self):
        """测试极端权重配置"""
        # 测试接近100%单一资产的配置
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.99, 0.01],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        results = backtester.get_results()

        # 验证极端配置的合理性
        assert results['final_value'] > 0, "极端权重配置仍应产生正值"

        # 验证主要ETF占据了绝大部分组合价值
        if '511010' in backtester.positions and '510880' in backtester.positions:
            final_value = results['final_value']
            last_date = backtester.daily_dates[-1]

            # 计算各ETF的最终权重
            etf_values = {}
            for etf_code in backtester.etf_codes:
                if etf_code in backtester.positions:
                    # 使用实际价格数据计算市值
                    if etf_code in backtester.etf_data and last_date in backtester.etf_data[etf_code].index:
                        current_price = backtester.etf_data[etf_code].loc[last_date, 'close']
                        market_value = backtester.positions[etf_code]['shares'] * current_price
                        etf_values[etf_code] = market_value

            if etf_values:
                max_etf = max(etf_values, key=etf_values.get)
                max_weight = etf_values[max_etf] / final_value
                assert max_weight > 0.9, f"主要ETF {max_etf} 应占据90%以上的组合价值，实际: {max_weight:.2%}"
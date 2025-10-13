"""
步骤4：再平衡功能测试
验证定期再平衡和阈值触发再平衡的正确性
"""

import pytest
import pandas as pd
import numpy as np
from portfolio_backtester import PortfolioBacktester


class TestRebalancing:
    """再平衡功能测试类"""

    def test_quarterly_rebalancing_dates(self, etf_data_dca):
        """测试季度再平衡日期计算"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'quarterly',
            'rebalance_threshold': 0.05
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证季度再平衡次数（应该在3、6、9、12月各一次）
        rebalance_count = backtester.get_results()['rebalance_count']
        assert 3 <= rebalance_count <= 4, \
            f"季度再平衡次数不合理: 期望3-4次，实际{rebalance_count}次"

        # 验证再平衡交易记录
        rebalance_transactions = [t for t in backtester.transactions
                                 if t['type'] in ['rebalance_buy', 'rebalance_sell']]
        assert len(rebalance_transactions) > 0, \
            "应该有再平衡交易记录"

    def test_monthly_rebalancing_dates(self, etf_data_dca):
        """测试月度再平衡日期计算"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'monthly',
            'rebalance_threshold': 0.1
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证月度再平衡次数
        rebalance_count = backtester.get_results()['rebalance_count']
        assert 5 <= rebalance_count <= 6, \
            f"月度再平衡次数不合理: 期望5-6次，实际{rebalance_count}次"

    def test_threshold_triggered_rebalancing(self, etf_data_dca):
        """测试阈值触发的再平衡"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'yearly',  # 很长的时间间隔，主要靠阈值触发
            'rebalance_threshold': 0.05  # 5%阈值，相对敏感
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 由于5%阈值相对敏感，应该有多次阈值触发的再平衡
        rebalance_count = backtester.get_results()['rebalance_count']
        assert rebalance_count >= 1, \
            f"阈值触发的再平衡应该至少发生1次，实际{rebalance_count}次"

        # 验证再平衡交易记录
        rebalance_transactions = [t for t in backtester.transactions
                                 if t['type'] in ['rebalance_buy', 'rebalance_sell']]
        assert len(rebalance_transactions) > 0, \
            "应该有再平衡交易记录"

    def test_rebalancing_execution(self, etf_data_dca):
        """测试再平衡执行逻辑"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.7, 0.3],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'monthly',
            'rebalance_threshold': 0.1
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证再平衡交易类型
        rebalance_buy_transactions = [t for t in backtester.transactions
                                     if t['type'] == 'rebalance_buy']
        rebalance_sell_transactions = [t for t in backtester.transactions
                                      if t['type'] == 'rebalance_sell']

        # 应该有买入和卖出交易
        assert len(rebalance_buy_transactions) > 0, "应该有再平衡买入交易"
        assert len(rebalance_sell_transactions) > 0, "应该有再平衡卖出交易"

        # 验证交易价格的合理性（应使用收盘价）
        for trans in rebalance_buy_transactions + rebalance_sell_transactions:
            date = trans['date']
            etf_code = trans['etf_code']
            price = trans['price']

            # 从缓存数据中获取对应日期的收盘价
            if etf_code in etf_data_dca and date in etf_data_dca[etf_code].index:
                expected_price = etf_data_dca[etf_code].loc[date, 'close']
                assert abs(price - expected_price) < 0.001, \
                    f"再平衡应使用收盘价: {date}, {etf_code}, 期望 {expected_price:.3f}, 实际 {price:.3f}"

    def test_rebalancing_weight_correction(self, etf_data_dca):
        """测试再平衡的权重修正效果"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'quarterly',
            'rebalance_threshold': 0.05
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 检查再平衡后的权重
        if backtester.rebalance_dates:
            # 获取最后一次再平衡后的权重
            last_rebalance_date = backtester.rebalance_dates[-1]

            # 找到最后一次再平衡后的第一个交易日
            post_rebalance_dates = [d for d in backtester.daily_dates if d > last_rebalance_date]
            if post_rebalance_dates:
                check_date = post_rebalance_dates[0]
                total_value = backtester._calculate_portfolio_value(check_date)

                # 验证权重是否接近目标权重
                for i, etf_code in enumerate(backtester.etf_codes):
                    if etf_code in backtester.positions and etf_code in etf_data_dca:
                        if check_date in etf_data_dca[etf_code].index:
                            current_price = etf_data_dca[etf_code].loc[check_date, 'close']
                            market_value = backtester.positions[etf_code]['shares'] * current_price
                            actual_weight = market_value / total_value
                            target_weight = backtester.weights[i]

                            # 再平衡后权重应该接近目标权重（允许小幅偏差）
                            assert abs(actual_weight - target_weight) < 0.02, \
                                f"再平衡后{etf_code}权重偏差过大: 目标 {target_weight:.2%}, 实际 {actual_weight:.2%}"

    def test_rebalancing_vs_no_rebalancing(self, etf_data_dca):
        """对比有再平衡和无再平衡的差异"""
        base_config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'enable_dca': False
        }

        # 无再平衡
        backtester_no_rebal = PortfolioBacktester(**base_config, enable_rebalancing=False)
        backtester_no_rebal.run_backtest()

        # 有再平衡
        backtester_with_rebal = PortfolioBacktester(**base_config, enable_rebalancing=True,
                                                   rebalance_freq='quarterly', rebalance_threshold=0.1)
        backtester_with_rebal.run_backtest()

        results_no_rebal = backtester_no_rebal.get_results()
        results_with_rebal = backtester_with_rebal.get_results()

        # 验证再平衡次数
        assert results_with_rebal['rebalance_count'] > 0, \
            "有再平衡的版本应该有再平衡记录"
        assert results_no_rebal['rebalance_count'] == 0, \
            "无再平衡的版本不应该有再平衡记录"

        # 验证再平衡对最终权重的影响
        final_value_no_rebal = results_no_rebal['final_value']
        final_value_with_rebal = results_with_rebal['final_value']

        print(f"无再平衡最终价值: ¥{final_value_no_rebal:,.2f}")
        print(f"有再平衡最终价值: ¥{final_value_with_rebal:,.2f}")
        print(f"再平衡次数: {results_with_rebal['rebalance_count']}")

        # 再平衡后的最终权重应该更接近目标权重
        def calculate_final_weights(backtester, etf_data):
            final_value = backtester.get_results()['final_value']
            last_date = backtester.daily_dates[-1]
            weights = {}

            for etf_code in backtester.etf_codes:
                if etf_code in backtester.positions and etf_code in etf_data:
                    if last_date in etf_data[etf_code].index:
                        current_price = etf_data[etf_code].loc[last_date, 'close']
                        market_value = backtester.positions[etf_code]['shares'] * current_price
                        weights[etf_code] = market_value / final_value
            return weights

        weights_no_rebal = calculate_final_weights(backtester_no_rebal, etf_data_dca)
        weights_with_rebal = calculate_final_weights(backtester_with_rebal, etf_data_dca)

        target_weights = {'511010': 0.6, '510880': 0.4}

        # 计算权重偏差
        deviation_no_rebal = sum(abs(weights_no_rebal.get(etf, 0) - target)
                                for etf, target in target_weights.items())
        deviation_with_rebal = sum(abs(weights_with_rebal.get(etf, 0) - target)
                                  for etf, target in target_weights.items())

        print(f"无再平衡权重偏差: {deviation_no_rebal:.3f}")
        print(f"有再平衡权重偏差: {deviation_with_rebal:.3f}")

        # 有再平衡的权重偏差应该更小（更接近目标权重）
        assert deviation_with_rebal <= deviation_no_rebal + 0.01, \
            "再平衡应该使组合权重更接近目标权重"

    def test_rebalancing_with_transaction_costs(self, etf_data_dca):
        """测试有交易成本的再平衡"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 100000,
            'transaction_cost': 0.003,  # 0.3%交易成本
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'monthly',
            'rebalance_threshold': 0.05
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证再平衡交易记录中包含了交易成本
        rebalance_transactions = [t for t in backtester.transactions
                                 if t['type'] in ['rebalance_buy', 'rebalance_sell']]

        assert len(rebalance_transactions) > 0, "应该有再平衡交易记录"

        # 验证交易成本的合理性
        for trans in rebalance_transactions:
            if trans['type'] == 'rebalance_buy':
                # 买入成本 = 股数 * 价格 * (1 + 交易成本)
                expected_cost = trans['shares'] * trans['price'] * (1 + config['transaction_cost'])
                assert abs(trans['amount'] - expected_cost) < 0.01, \
                    f"买入交易成本计算错误: {trans}"
            elif trans['type'] == 'rebalance_sell':
                # 卖出收入 = 股数 * 价格 * (1 - 交易成本)
                expected_proceeds = trans['shares'] * trans['price'] * (1 - config['transaction_cost'])
                assert abs(trans['amount'] - expected_proceeds) < 0.01, \
                    f"卖出交易成本计算错误: {trans}"

    def test_rebalancing_with_dca(self, etf_data_dca):
        """测试再平衡与定投同时使用"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 50000,
            'transaction_cost': 0.001,
            'enable_dca': True,
            'dca_amount': 5000,
            'dca_freq': 'monthly',
            'enable_rebalancing': True,
            'rebalance_freq': 'quarterly',
            'rebalance_threshold': 0.1
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        results = backtester.get_results()

        # 验证定投和再平衡都发生了
        assert results['dca_count'] > 0, "应该有定投记录"
        assert results['rebalance_count'] > 0, "应该有再平衡记录"

        # 验证总投入计算
        expected_total_investment = config['initial_capital'] + config['dca_amount'] * results['dca_count']
        assert abs(results['total_investment'] - expected_total_investment) < 1.0, \
            f"定投+再平衡总投入计算错误: 期望 {expected_total_investment}, 实际 {results['total_investment']}"

        # 验证交易记录类型
        transaction_types = set(t['type'] for t in backtester.transactions)
        expected_types = {'initial_buy', 'dca_buy', 'rebalance_buy', 'rebalance_sell'}

        # 检查是否包含预期的交易类型
        assert len(transaction_types & expected_types) >= 3, \
            f"交易记录应该包含多种类型: 实际 {transaction_types}"

    def test_rebalancing_threshold_sensitivity(self):
        """测试再平衡阈值敏感性"""
        configs = [
            {'threshold': 0.02, 'description': '敏感阈值(2%)'},
            {'threshold': 0.10, 'description': '宽松阈值(10%)'},
            {'threshold': 0.20, 'description': '很宽松阈值(20%)'}
        ]

        results = {}
        base_config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.6, 0.4],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 100000,
            'transaction_cost': 0,
            'enable_dca': False,
            'enable_rebalancing': True,
            'rebalance_freq': 'yearly'  # 主要靠阈值触发
        }

        for config_update in configs:
            config = base_config.copy()
            config['rebalance_threshold'] = config_update['threshold']

            backtester = PortfolioBacktester(**config)
            backtester.run_backtest()

            results[config_update['description']] = {
                'rebalance_count': backtester.get_results()['rebalance_count'],
                'final_value': backtester.get_results()['final_value']
            }

        print("不同阈值的再平衡结果:")
        for desc, result in results.items():
            print(f"  {desc}: 再平衡{result['rebalance_count']}次, 最终价值¥{result['final_value']:,.2f}")

        # 验证阈值敏感性的合理性
        # 更敏感的阈值应该产生更多的再平衡
        sensitive_count = results['敏感阈值(2%)']['rebalance_count']
        loose_count = results['宽松阈值(10%)']['rebalance_count']
        very_loose_count = results['很宽松阈值(20%)']['rebalance_count']

        assert sensitive_count >= loose_count, \
            "更敏感的阈值应该产生更多或相等的再平衡次数"
        assert loose_count >= very_loose_count, \
            "更宽松的阈值应该产生更少或相等的再平衡次数"
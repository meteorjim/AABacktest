"""
步骤2：定投（DCA）功能测试
使用独立的akshare数据获取和计算逻辑，验证定投功能
"""

import pytest
import pandas as pd
import numpy as np
import akshare as ak
from portfolio_backtester import PortfolioBacktester


class TestDCA:
    """定投功能测试类"""

    def test_dca_dates_monthly(self, dca_monthly_config, etf_data_dca):
        """测试月度定投日期计算"""
        # 使用fixtures提供的缓存数据进行日期验证
        etf_code = dca_monthly_config['etf_codes'][0]

        if etf_code not in etf_data_dca:
            pytest.skip(f"ETF {etf_code} 缓存数据不可用")

        data = etf_data_dca[etf_code].copy()

        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 独立计算定投日期 - 匹配PortfolioBacktester的逻辑
        data.index = pd.to_datetime(data.index)
        trading_dates = sorted(data.index.tolist())

        # 跳过第一个交易日（初始建仓日）
        trading_dates_for_dca = trading_dates[1:]

        # 计算每月最后一个交易日
        expected_dca_dates = []
        current_month = None
        processed_months = set()

        for date in trading_dates_for_dca:
            if date.month not in processed_months:
                # 找到这个月的所有交易日
                month_dates = [d for d in trading_dates_for_dca
                             if d.month == date.month and d.year == date.year]
                if month_dates:
                    expected_dca_dates.append(month_dates[-1])  # 月末最后一个交易日
                    processed_months.add(date.month)

        # 验证定投日期
        actual_dca_dates = backtester.dca_dates

        assert len(actual_dca_dates) == len(expected_dca_dates), \
            f"定投次数不匹配: 期望 {len(expected_dca_dates)}, 实际 {len(actual_dca_dates)}"

        # 验证每个定投日期
        for i, (expected, actual) in enumerate(zip(expected_dca_dates, actual_dca_dates)):
            assert expected == actual, \
                f"第{i+1}次定投日期错误: 期望 {expected.strftime('%Y-%m-%d')}, 实际 {actual.strftime('%Y-%m-%d')}"

    def test_dca_buy_execution(self, dca_monthly_config, etf_data_dca):
        """测试定投买入执行"""
        # 使用fixtures提供的缓存数据
        etf_codes = dca_monthly_config['etf_codes']
        weights = dca_monthly_config['weights']

        # 检查所有ETF数据是否可用
        for etf_code in etf_codes:
            if etf_code not in etf_data_dca:
                pytest.skip(f"ETF {etf_code} 缓存数据不可用")

        etf_data = {}
        for etf_code in etf_codes:
            data = etf_data_dca[etf_code].copy()
            data.index = pd.to_datetime(data.index)
            etf_data[etf_code] = data

        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 验证每次定投交易
        dca_transactions = [t for t in backtester.transactions
                          if t['type'] == 'dca_buy']

        assert len(dca_transactions) > 0, \
            "应该有定投交易记录"

        for trans in dca_transactions:
            date = trans['date']
            etf_code = trans['etf_code']

            # 获取当天的收盘价（独立数据源）
            data = etf_data[etf_code]

            assert date in data.index, \
                f"日期 {date} 在独立数据中找不到"

            close_price = data.loc[date, 'close']

            # 验证使用收盘价买入
            assert abs(trans['price'] - close_price) < 0.001, \
                f"定投未使用收盘价: 日期 {date.strftime('%Y-%m-%d')}, 期望 {close_price:.3f}, 实际 {trans['price']:.3f}"

            # 验证买入股数计算
            dca_amount = dca_monthly_config['dca_amount']
            etf_weight = weights[etf_codes.index(etf_code)]
            target_value = dca_amount * etf_weight
            # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
            expected_shares = target_value / (close_price * (1 + backtester.transaction_cost))

            assert abs(trans['shares'] - expected_shares) < 0.001, \
                f"定投股数计算错误: 期望 {expected_shares:.4f}, 实际 {trans['shares']:.4f}"

    def test_dca_average_cost_update(self, dca_monthly_config):
        """测试定投后平均成本更新"""
        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 独立重新计算平均成本
        for etf_code in backtester.positions:
            position = backtester.positions[etf_code]

            # 获取所有该ETF的交易
            etf_transactions = [t for t in backtester.transactions
                              if t['etf_code'] == etf_code]

            # 计算加权平均成本
            total_shares = 0
            total_cost = 0

            for trans in etf_transactions:
                total_shares += trans['shares']
                total_cost += trans['amount']

            expected_avg_cost = total_cost / total_shares if total_shares > 0 else 0

            # 验证平均成本
            assert abs(position['avg_cost'] - expected_avg_cost) < 0.001, \
                f"{etf_code} 平均成本错误: 期望 {expected_avg_cost:.3f}, 实际 {position['avg_cost']:.3f}"

            # 验证总股数
            assert abs(position['shares'] - total_shares) < 0.001, \
                f"{etf_code} 总股数错误: 期望 {total_shares:.4f}, 实际 {position['shares']:.4f}"

            # 验证总成本
            assert abs(position['total_cost'] - total_cost) < 1.0, \
                f"{etf_code} 总成本错误: 期望 {total_cost:.2f}, 实际 {position['total_cost']:.2f}"

    def test_dca_total_investment(self, dca_monthly_config):
        """测试定投总投入计算"""
        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 独立计算总投入
        expected_total = (dca_monthly_config['initial_capital'] +
                         dca_monthly_config['dca_amount'] * len(backtester.dca_dates))

        actual_total = backtester.get_results()['total_investment']

        # 验证总投入
        assert abs(actual_total - expected_total) < 1.0, \
            f"总投入计算错误: 期望 {expected_total:.2f}, 实际 {actual_total:.2f}"

    def test_dca_return_calculation(self, dca_monthly_config):
        """测试定投收益率计算"""
        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 独立计算收益率
        results = backtester.get_results()
        total_investment = results['total_investment']
        final_value = results['final_value']

        expected_return = (final_value / total_investment - 1) * 100
        actual_return = results['total_return']

        # 验证收益率
        assert abs(actual_return - expected_return) < 0.01, \
            f"定投收益率计算错误: 期望 {expected_return:.2f}%, 实际 {actual_return:.2f}%"

    def test_dca_vs_no_dca_comparison(self, etf_data_single):
        """对比定投和非定投的差异"""
        base_config = {
            'etf_codes': ['511010'],
            'weights': [1.0],
            'start_date': '2024-01-01',
            'end_date': '2024-06-30',
            'initial_capital': 50000,
            'transaction_cost': 0
        }

        # 使用fixtures提供的缓存数据
        etf_code = base_config['etf_codes'][0]

        if etf_code not in etf_data_single:
            pytest.skip(f"ETF {etf_code} 缓存数据不可用")

        # 不使用定投
        backtester_no_dca = PortfolioBacktester(**base_config, enable_dca=False)
        backtester_no_dca.run_backtest()

        # 使用定投
        dca_config = base_config.copy()
        dca_config.update({
            'enable_dca': True,
            'dca_amount': 10000,
            'dca_freq': 'monthly'
        })
        backtester_dca = PortfolioBacktester(**dca_config)
        backtester_dca.run_backtest()

        # 获取结果
        results_no_dca = backtester_no_dca.get_results()
        results_dca = backtester_dca.get_results()

        # 验证定投版本的总投入更高
        assert results_dca['total_investment'] > results_no_dca['total_investment'], \
            "定投版本的总投入应该更高"

        # 验证定投次数合理
        expected_months = 6  # 1月到6月
        actual_dca_count = len(backtester_dca.dca_dates)
        assert abs(actual_dca_count - expected_months) <= 1, \
            f"定投次数不合理: 期望约 {expected_months}, 实际 {actual_dca_count}"

        # 验证定投总投入
        expected_total_investment = base_config['initial_capital'] + 10000 * actual_dca_count
        assert abs(results_dca['total_investment'] - expected_total_investment) < 1.0, \
            f"定投总投入计算错误: 期望 {expected_total_investment}, 实际 {results_dca['total_investment']}"

    def test_dca_cash_flow(self, dca_monthly_config):
        """测试定投现金流"""
        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 验证最终现金接近0
        final_cash = backtester.cash
        assert final_cash >= 0, \
            f"最终现金不能为负: {final_cash:.2f}"

        # 验证现金利用率
        if backtester.dca_dates:
            # 定投后现金应该很少
            cash_ratio = final_cash / backtester.get_results()['total_investment']
            assert cash_ratio < 0.01, \
                f"定投后现金余额比例过高: {cash_ratio:.4f}"

    @pytest.mark.parametrize("dca_amount", [1000, 5000, 10000])
    def test_different_dca_amounts(self, dca_amount):
        """测试不同定投金额"""
        config = {
            'etf_codes': ['511010'],
            'weights': [1.0],
            'start_date': '2024-01-01',
            'end_date': '2024-03-31',
            'initial_capital': 10000,
            'transaction_cost': 0,
            'enable_dca': True,
            'dca_amount': dca_amount,
            'dca_freq': 'monthly'
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证定投执行
        dca_transactions = [t for t in backtester.transactions
                          if t['type'] == 'dca_buy']

        assert len(dca_transactions) > 0, \
            f"定投金额 {dca_amount} 应该有定投交易"

        # 验证每次定投金额
        for trans in dca_transactions:
            # 对于单只ETF，定投交易金额应该等于定投金额
            # 因为目标价值 = dca_amount，而实际成本 = shares * price * (1 + cost)
            # 这应该约等于dca_amount
            assert abs(trans['amount'] - dca_amount) < 1.0, \
                f"定投交易金额错误: 期望 {dca_amount}, 实际 {trans['amount']:.2f}"

    def test_dca_multiple_etf_weights(self, dca_monthly_config, etf_data_dca):
        """测试多ETF定投的权重分配"""
        # 使用fixtures提供的缓存数据
        etf_codes = dca_monthly_config['etf_codes']
        weights = dca_monthly_config['weights']

        # 检查所有ETF数据是否可用
        for etf_code in etf_codes:
            if etf_code not in etf_data_dca:
                pytest.skip(f"ETF {etf_code} 缓存数据不可用")

        etf_data = {}
        for etf_code in etf_codes:
            data = etf_data_dca[etf_code].copy()
            data.index = pd.to_datetime(data.index)
            etf_data[etf_code] = data

        # 运行回测
        backtester = PortfolioBacktester(**dca_monthly_config)
        backtester.run_backtest()

        # 获取每次定投的分配
        dca_groups = {}
        for trans in backtester.transactions:
            if trans['type'] == 'dca_buy':
                date = trans['date']
                if date not in dca_groups:
                    dca_groups[date] = {}
                dca_groups[date][trans['etf_code']] = trans

        # 验证每次定投的权重分配
        for date, transactions in dca_groups.items():
            total_amount = sum(t['amount'] for t in transactions.values())

            for etf_code, trans in transactions.items():
                weight_index = etf_codes.index(etf_code)
                expected_weight = weights[weight_index]
                actual_weight = trans['amount'] / total_amount

                # 允许小幅误差（交易成本影响）
                assert abs(actual_weight - expected_weight) < 0.01, \
                    f"日期 {date.strftime('%Y-%m-%d')} ETF {etf_code} 权重分配错误: " \
                    f"期望 {expected_weight:.2f}, 实际 {actual_weight:.2f}"

    def test_dca_price_validation(self, etf_data_single):
        """验证定投使用收盘价"""
        # 短期测试便于验证
        config = {
            'etf_codes': ['511010'],
            'weights': [1.0],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'initial_capital': 10000,
            'transaction_cost': 0,
            'enable_dca': True,
            'dca_amount': 5000,
            'dca_freq': 'monthly'
        }

        # 使用fixtures提供的缓存数据
        etf_code = config['etf_codes'][0]

        if etf_code not in etf_data_single:
            pytest.skip(f"ETF {etf_code} 缓存数据不可用")

        data = etf_data_single[etf_code].copy()
        data.index = pd.to_datetime(data.index)

        # 运行回测
        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证定投交易使用收盘价
        dca_transactions = [t for t in backtester.transactions
                          if t['type'] == 'dca_buy']

        for trans in dca_transactions:
            date = trans['date']

            assert date in data.index, \
                f"日期 {date} 在独立数据中找不到"

            open_price = data.loc[date, 'open']
            close_price = data.loc[date, 'close']
            high_price = data.loc[date, 'high']
            low_price = data.loc[date, 'low']

            # 验证定投使用收盘价
            assert abs(trans['price'] - close_price) < 0.001, \
                f"定投未使用收盘价: 日期 {date.strftime('%Y-%m-%d')}, " \
                f"开盘 {open_price:.3f}, 收盘 {close_price:.3f}, 买入 {trans['price']:.3f}"

            # 验证不是使用其他价格
            assert abs(trans['price'] - open_price) > 0.01, \
                "定投不应该使用开盘价"

    @pytest.mark.slow
    def test_dca_long_period(self):
        """测试长期定投"""
        config = {
            'etf_codes': ['511010', '510880'],
            'weights': [0.5, 0.5],
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',  # 2年
            'initial_capital': 50000,
            'transaction_cost': 0,
            'enable_dca': True,
            'dca_amount': 2000,
            'dca_freq': 'monthly'
        }

        backtester = PortfolioBacktester(**config)
        backtester.run_backtest()

        # 验证定投次数合理
        expected_dca_count = 24  # 2年 × 12月
        actual_dca_count = len(backtester.dca_dates)

        # 允许一定的误差（可能某些月份无交易日）
        assert abs(actual_dca_count - expected_dca_count) <= 2, \
            f"长期定投次数不合理: 期望约 {expected_dca_count}, 实际 {actual_dca_count}"

        # 验证最终持仓合理
        results = backtester.get_results()
        assert results['final_value'] > 0, \
            "长期定投最终价值应大于0"

        # 验证持仓包含两只ETF
        assert len(backtester.positions) == 2, \
            "应该持有两只ETF"

        # 验证总投入计算正确
        expected_total = config['initial_capital'] + config['dca_amount'] * actual_dca_count
        assert abs(results['total_investment'] - expected_total) < 1.0, \
            f"长期定投总投入计算错误"
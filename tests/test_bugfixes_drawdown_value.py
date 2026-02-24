import pytest
import pandas as pd
import numpy as np
from portfolio_backtester import PortfolioBacktester
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_dca_drawdown_data():
    """造一个定投场景导致旧算式回撤爆炸，但由于TWR实际回撤并不大的假数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    # ETF 价格：先涨，中间微跌，后继续涨。制造轻微回撤
    # Base: 100 -> 110 -> 105 (-4.5%) -> 115 -> 120
    prices = [100.0, 105.0, 110.0, 108.0, 105.0, 108.0, 112.0, 115.0]
    dates = dates[:len(prices)]
    
    df = pd.DataFrame({
        'open': prices,
        'close': prices,
        'high': [p + 1 for p in prices],
        'low': [p - 1 for p in prices],
        'volume': [1000] * len(prices)
    }, index=dates[:len(prices)])
    
    return {'TEST_ETF': df}

@patch('portfolio_backtester.PortfolioBacktester.fetch_data')
def test_twr_max_drawdown_with_dca(mock_fetch_data, mock_dca_drawdown_data):
    """验证定投过程中的 TWR 回撤计算是否正常，而不是基于绝对数值偏差爆炸"""
    # 模拟数据加载
    mock_fetch_data.return_value = None
    
    backtester = PortfolioBacktester(
        etf_codes=['TEST_ETF'],
        weights=[100],
        start_date='2023-01-01',
        end_date='2023-01-10',
        initial_capital=10000,
        enable_dca=True,
        dca_amount=500000, # 极大的定投金额以扭曲传统绝对值回撤
        dca_freq='monthly',
        force_refresh=False,
        save_html=False
    )
    
    backtester.etf_data = mock_dca_drawdown_data
    
    # 手动让它在中间某个交易日发生巨大定投
    original_get_dca_dates = backtester._get_dca_dates
    def mock_get_dca_dates(trading_dates):
        # 挑选价格高点（第3天: 110.0）的日子进行定投
        return [trading_dates[2]]
    
    backtester._get_dca_dates = mock_get_dca_dates
    
    backtester.run_backtest()
    
    # 理论 TWR 回撤应该是从 110 跌到 105 = -4.54%
    # 如果是基于绝对数值或者积累旧逻辑计算，巨量资金在 110 涌入后跌到 105 损失极大，旧的百分点相减法更可能超过 -100%
    expected_drawdown = (105.0 - 110.0) / 110.0 * 100
    
    actual_drawdown = backtester.results['max_drawdown']
    
    # 允许由于滑点和精度造成的极微小偏差
    assert actual_drawdown < 0, "最大回撤应该是负数"
    assert actual_drawdown > -10.0, "回撤不应该被巨量入金定投扭曲放大到甚至跌破-100% (应为-4.5%左右)"
    assert round(actual_drawdown, 1) == round(expected_drawdown, 1)

@pytest.fixture
def mock_missing_start_data():
    """造一组假数据，两个ETF，一个在建仓日有数据，另一个不仅没数据而且之前也没数据"""
    dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
    
    # ETF_A 在 1月1日(建仓)就有数据
    df_a = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5]
    }, index=dates)
    
    # ETF_B 在 1月1日和2日都没有数据，从3日才开始有
    df_b = pd.DataFrame({
        'open': [50.0, 51.0, 52.0],
        'close': [50.5, 51.5, 52.5]
    }, index=dates[2:])
    
    return {'ETF_A': df_a, 'ETF_B': df_b}

@patch('portfolio_backtester.PortfolioBacktester.fetch_data')
def test_initial_value_with_missing_data(mock_fetch_data, mock_missing_start_data):
    """验证开局由于节假日时差导致某ETF无记录时，每日投资组合价值不会陡降"""
    mock_fetch_data.return_value = None
    
    backtester = PortfolioBacktester(
        etf_codes=['ETF_A', 'ETF_B'],
        weights=[50, 50],
        start_date='2023-01-01',
        end_date='2023-01-05',
        initial_capital=100000,
        enable_dca=False,
        force_refresh=False,
        save_html=False
    )
    
    backtester.etf_data = mock_missing_start_data
    backtester.run_backtest()
    
    # 在 2023-01-01 (第一天):
    # ETF_A 有价格。建仓花费约 50000，余现金约 50000
    # ETF_B 无价格，等 2023-01-03 才买。此时现金留存
    
    # 所有日期的 portfolio_value 都不应该比初始资本插水（旧 Bug 会在第一天估值为 5w）
    for i, date in enumerate(backtester.daily_dates):
        value = backtester.daily_values[i]
        # 第一天价值应该是 100000 左右 (加上 ETF_A 当日的浮盈)
        # 用 > 95000 判断，证明 50000 的资产没有被清零
        assert value > 99000, f"在 {date} 的资产估值 {value} 异常，疑似资产蒸发"

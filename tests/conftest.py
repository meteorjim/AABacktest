"""
pytest配置文件
定义共享的fixtures供所有测试使用
使用data目录的缓存数据和get_price_akshare函数，确保数据获取稳定可靠
"""

import pytest
import sys
import os

# 添加父目录到路径，以便导入PortfolioBacktester和get_price_akshare
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_backtester import PortfolioBacktester, get_price_akshare


@pytest.fixture(scope="session")
def etf_data_single():
    """使用缓存数据和get_price_akshare获取单只ETF测试数据"""
    etf_code = '511010'
    start_date = '2024-01-01'
    end_date = '2024-03-31'

    try:
        # 使用get_price_akshare函数（模块级函数，不属于PortfolioBacktester类）
        # 优先使用缓存数据，确保测试稳定性
        data = get_price_akshare(
            stock=etf_code,
            start_date=start_date,
            end_date=end_date,
            need_ma=False,
            use_cache=True,
            cache_dir='data',
            force_refresh=False
        )

        if data.empty:
            print(f"警告: {etf_code} 在 {start_date} 到 {end_date} 期间无数据")
            return {}

        print(f"使用缓存数据获取 {etf_code}: {len(data)} 条记录")
        return {etf_code: data}

    except Exception as e:
        print(f"获取 {etf_code} 数据失败: {e}")
        return {}


@pytest.fixture(scope="session")
def etf_data_multiple():
    """使用缓存数据和get_price_akshare获取多只ETF测试数据"""
    etf_codes = ['511010', '510880']
    start_date = '2024-01-01'
    end_date = '2024-03-31'

    data = {}
    for etf_code in etf_codes:
        try:
            # 使用get_price_akshare函数（模块级函数，不属于PortfolioBacktester类）
            df = get_price_akshare(
                stock=etf_code,
                start_date=start_date,
                end_date=end_date,
                need_ma=False,
                use_cache=True,
                cache_dir='data',
                force_refresh=False
            )

            if not df.empty:
                data[etf_code] = df
                print(f"使用缓存数据获取 {etf_code}: {len(df)} 条记录")
            else:
                print(f"警告: {etf_code} 在 {start_date} 到 {end_date} 期间无数据")

        except Exception as e:
            print(f"获取 {etf_code} 数据失败: {e}")
            continue

    return data


@pytest.fixture(scope="session")
def etf_data_dca():
    """使用缓存数据和get_price_akshare获取定投测试数据（覆盖更长时间）"""
    etf_codes = ['511010', '510880']
    start_date = '2024-01-01'
    end_date = '2024-06-30'

    data = {}
    for etf_code in etf_codes:
        try:
            # 使用get_price_akshare函数（模块级函数，不属于PortfolioBacktester类）
            df = get_price_akshare(
                stock=etf_code,
                start_date=start_date,
                end_date=end_date,
                need_ma=False,
                use_cache=True,
                cache_dir='data',
                force_refresh=False
            )

            if not df.empty:
                data[etf_code] = df
                print(f"使用缓存数据获取 {etf_code}: {len(df)} 条记录")
            else:
                print(f"警告: {etf_code} 在 {start_date} 到 {end_date} 期间无数据")

        except Exception as e:
            print(f"获取 {etf_code} 数据失败: {e}")
            continue

    return data


@pytest.fixture
def single_etf_config():
    """单只ETF回测配置"""
    return {
        'etf_codes': ['511010'],
        'weights': [1.0],
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',
        'initial_capital': 100000,
        'transaction_cost': 0,
        'enable_dca': False
    }


@pytest.fixture
def multiple_etf_config():
    """多只ETF回测配置"""
    return {
        'etf_codes': ['511010', '510880'],
        'weights': [0.3, 0.7],
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'enable_dca': False
    }


@pytest.fixture
def dca_monthly_config():
    """月度定投配置"""
    return {
        'etf_codes': ['511010', '510880'],
        'weights': [0.5, 0.5],
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'initial_capital': 20000,
        'transaction_cost': 0,
        'enable_dca': True,
        'dca_amount': 5000,
        'dca_freq': 'monthly'
    }


@pytest.fixture
def backtester_single(single_etf_config):
    """单只ETF回测器实例"""
    return PortfolioBacktester(**single_etf_config)


@pytest.fixture
def backtester_multiple(multiple_etf_config):
    """多只ETF回测器实例"""
    return PortfolioBacktester(**multiple_etf_config)


@pytest.fixture
def backtester_dca(dca_monthly_config):
    """定投回测器实例"""
    return PortfolioBacktester(**dca_monthly_config)


@pytest.fixture(autouse=True)
def cleanup_cache():
    """测试后清理缓存（可选）"""
    yield
    # 这里可以添加清理逻辑，如果需要的话
    pass


def pytest_configure(config):
    """pytest配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 自动标记慢速测试
    for item in items:
        if "dca" in item.nodeid.lower() and "monthly" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
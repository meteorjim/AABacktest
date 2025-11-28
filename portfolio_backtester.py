import pandas as pd
import numpy as np
import datetime
import akshare as ak
from typing import List, Dict, Optional
import os

# 图表相关导入
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 设置pandas选项以避免未来警告
pd.set_option('future.no_silent_downcasting', True)


def safe_pct_change(series: pd.Series, fillna: bool = True) -> pd.Series:
    """
    安全的百分比变化计算，处理缺失数据

    Args:
        series: 价格序列
        fillna: 是否用0填充缺失的收益率

    Returns:
        pd.Series: 收益率序列，缺失值用0填充
    """
    returns = series.pct_change()
    # 将inf和-inf替换为NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)

    if fillna:
        # 对于缺失的数据，假设价格没有变化，收益率为0
        returns = returns.fillna(0)

    return returns


def get_price_akshare(stock: str, start_date, end_date, need_ma=True,
                     use_cache=True, cache_dir='data', force_refresh=False) -> pd.DataFrame:
    """
    使用akshare接口获取A股历史价格数据，支持本地缓存

    Parameters:
        stock (str): A股代码（如000001.SZ, 600000.SH）
        start_date (str): 开始日期
        end_date (str): 结束日期
        need_ma (bool): 是否计算移动平均线
        use_cache (bool): 是否使用本地缓存，默认True
        cache_dir (str): 缓存目录，默认'data'
        force_refresh (bool): 是否强制刷新缓存，默认False

    Returns:
        pandas.DataFrame: 历史价格数据，包含列：
            - close: 收盘价
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - volume: 成交量
            - cr: 涨跌幅
            - MA5, MA13, MA21, MA60: 移动平均线（如果need_ma=True）
    """

    def _get_cache_file_path(stock, start_date, end_date, cache_dir):
        """生成缓存文件路径"""
        start_str = start_date.replace('-', '') if isinstance(start_date, str) else start_date.strftime('%Y%m%d')
        end_str = end_date.replace('-', '') if isinstance(end_date, str) else end_date.strftime('%Y%m%d')
        filename = f"{stock}_{start_str}_{end_str}.csv"
        return os.path.join(cache_dir, filename)

    def _find_matching_cache_file(stock, start_date, end_date, cache_dir):
        """查找包含所需时间范围的现有缓存文件"""
        try:
            if not os.path.exists(cache_dir):
                return None

            target_start = pd.to_datetime(start_date)
            target_end = pd.to_datetime(end_date)

            # 查找所有相关的缓存文件
            cache_files = []
            for filename in os.listdir(cache_dir):
                if filename.startswith(f"{stock}_") and filename.endswith(".csv"):
                    cache_files.append(filename)

            # 按文件名排序（包含时间范围更大的文件优先）
            cache_files.sort(reverse=True)

            for filename in cache_files:
                try:
                    # 解析文件名中的时间范围
                    parts = filename.replace('.csv', '').split('_')
                    if len(parts) >= 3:
                        file_start = pd.to_datetime(parts[1], format='%Y%m%d')
                        file_end = pd.to_datetime(parts[2], format='%Y%m%d')

                        # 检查缓存文件是否完全包含所需时间范围
                        if file_start <= target_start and file_end >= target_end:
                            return os.path.join(cache_dir, filename)
                except Exception as e:
                    print(f"解析缓存文件名失败 {filename}: {e}")
                    continue
            return None
        except Exception as e:
            print(f"查找匹配缓存文件时出错: {e}")
            return None

    def _save_to_cache(data, file_path):
        """保存数据到缓存文件"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data.to_csv(file_path, encoding='utf-8-sig')
            print(f"数据已缓存到: {file_path}")
        except Exception as e:
            print(f"保存缓存文件失败: {e}")

    def _load_from_cache(file_path):
        """从缓存文件加载数据"""
        try:
            if os.path.exists(file_path):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return data
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
        return None

    try:
        # 确保日期格式正确
        start_str = start_date.strftime('%Y%m%d') if isinstance(start_date, datetime.datetime) else start_date.replace('-', '')
        end_str = end_date.strftime('%Y%m%d') if isinstance(end_date, datetime.datetime) else end_date.replace('-', '')

        # 将A股代码转换为akshare格式（去掉后缀）
        symbol = stock.split('.')[0]

        # 缓存文件路径
        cache_file_path = _get_cache_file_path(stock, start_date, end_date, cache_dir)

        # 尝试从缓存加载数据（如果启用缓存且不强制刷新）
        data = None
        if use_cache and not force_refresh:
            # 首先尝试智能匹配现有缓存文件
            matching_cache = _find_matching_cache_file(stock, start_date, end_date, cache_dir)
            if matching_cache:
                data = _load_from_cache(matching_cache)
                if data is not None and not data.empty:
                    # 从缓存数据中筛选所需时间范围
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    filtered_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                    return filtered_data

        # 从akshare获取历史数据
        # 方法1: 尝试获取股票数据
        try:
            stock_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="hfq")
            if stock_df is not None and not stock_df.empty:
                data = stock_df
                print(f"找到了{stock} 的A股数据")
        except:
            pass

        # 方法2: 尝试获取ETF数据
        if data is None:
            try:
                etf_df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="hfq")
                if etf_df is not None and not etf_df.empty:
                    data = etf_df
                    print(f"找到了{stock} 的ETF数据")
            except:
                pass

        # 方法3: 尝试获取指数数据
        if data is None:
            try:
                index_df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str)
                if index_df is not None and not index_df.empty:
                    data = index_df
                    print(f"找到了{stock} 的指数数据")
            except:
                pass

        if data is None or data.empty:
            print(f"警告: 未能获取到 {stock} 的数据")
            return pd.DataFrame()

        # 重命名列以匹配原有格式
        data = data.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        })

        # 确保日期索引格式正确
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')

        # 计算涨跌幅
        data = data.sort_index()
        data['cr'] = safe_pct_change(data['close'], fillna=True)

        # 计算移动平均线
        if need_ma:
            data['MA5'] = data['close'].rolling(5).mean()
            data['MA13'] = data['close'].rolling(13).mean()
            data['MA21'] = data['close'].rolling(21).mean()
            data['MA60'] = data['close'].rolling(60).mean()
            result = data[['close', 'open', 'high', 'low', 'volume', 'cr', 'MA5', 'MA13', 'MA21', 'MA60']].dropna(subset=['cr'])
        else:
            result = data[['close', 'open', 'high', 'low', 'volume', 'cr']].dropna(subset=['cr'])

        # 保存到缓存（如果启用缓存且数据不为空）
        if use_cache and result is not None and not result.empty:
            _save_to_cache(result, cache_file_path)

        # print(f"成功获取 {stock} 数据，共 {len(result)} 条记录")
        return result

    except Exception as e:
        print(f"获取 {stock} 数据时出错: {e}")
        # 如果akshare失败，尝试从缓存加载（如果启用缓存）
        if use_cache and not force_refresh:
            print(f"尝试从缓存加载 {stock} 数据...")
            cached_data = _load_from_cache(cache_file_path)
            if cached_data is not None and not cached_data.empty:
                return cached_data
        return pd.DataFrame()


class PortfolioBacktester:
    """
    资产配置策略回测系统（重构版）

    功能：
    - 支持多ETF资产配置回测
    - 支持定期再平衡
    - 支持定投（DCA）
    - 计算各项性能指标

    策略优化：
    - 采用"先平衡后定投"执行顺序，减少逻辑冲突
    - 避免定投买入后立即卖出的情况
    - 降低交易成本，提升投资效率
    """

    def __init__(self,
                 etf_codes: List[str],
                 weights: List[float],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.000,
                 enable_dca: bool = False,
                 dca_amount: float = 10000,
                 dca_freq: str = 'monthly',
                 enable_rebalancing: bool = False,
                 rebalance_freq: str = 'quarterly',
                 rebalance_threshold: float = 0.0,
                 risk_free_rate: float = 0.02,
                 force_refresh: bool = False,
                 verbose_trading: bool = False,
                 save_html: bool = False):
        """
        初始化回测器

        Args:
            etf_codes: ETF代码列表
            weights: 对应权重列表（总和应为1）
            start_date: 回测开始日期 (格式: 'YYYY-MM-DD')
            end_date: 回测结束日期 (格式: 'YYYY-MM-DD')
            initial_capital: 初始资金
            transaction_cost: 交易成本比例
            enable_dca: 是否启用定投
            dca_amount: 定投金额
            dca_freq: 定投频率 ('monthly', 'yearly')
            enable_rebalancing: 是否启用再平衡
            rebalance_freq: 再平衡频率 ('monthly', 'quarterly', 'yearly')
            rebalance_threshold: 再平衡触发阈值（权重偏离超过此值时触发）
            risk_free_rate: 无风险利率，默认3% (0.03)
            force_refresh: 是否强制刷新缓存数据，默认False
            verbose_trading: 是否显示详细的定投和再平衡交易信息，默认False
            save_html: 是否将图表保存为HTML文件，默认False
        """
        self.etf_codes = etf_codes
        self.weights = np.array(weights)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # 定投相关参数
        self.enable_dca = enable_dca
        self.dca_amount = dca_amount
        self.dca_freq = dca_freq

        # 再平衡相关参数
        self.enable_rebalancing = enable_rebalancing
        self.rebalance_freq = rebalance_freq
        self.rebalance_threshold = rebalance_threshold

        # 无风险利率参数
        self.risk_free_rate = risk_free_rate

        # 数据刷新参数
        self.force_refresh = force_refresh

        # 交易详细打印参数
        self.verbose_trading = verbose_trading

        # HTML保存参数
        self.save_html = save_html

        # 验证权重
        if abs(np.sum(weights) - 1.0) > 0.001:
            raise ValueError("权重总和必须等于1")
        if len(etf_codes) != len(weights):
            raise ValueError("ETF代码数量与权重数量不匹配")

        # 初始化数据存储
        self.etf_data = {}  # ETF价格数据
        self.positions = {}  # 持仓信息 {etf_code: {'shares': float, 'avg_cost': float}}
        self.cash = initial_capital  # 现金

        # 历史记录
        self.daily_values = []  # 每日组合价值
        self.daily_dates = []  # 对应日期
        self.daily_positions = {}  # 每日ETF持仓详情 {date: {etf_code: shares, value, weight}}
        self.transactions = []  # 交易记录
        self.dca_dates = []  # 定投日期
        self.rebalance_dates = []  # 再平衡日期
        self.daily_flows = {}  # 每日现金流 {date: amount}

        # 回测结果
        self.results = {}

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有ETF的价格数据"""
        for etf_code in self.etf_codes:
            data = get_price_akshare(
                stock=etf_code,
                start_date=self.start_date,
                end_date=self.end_date,
                need_ma=False,
                use_cache=True,
                cache_dir='data',
                force_refresh=self.force_refresh
            )

            if data.empty:
                print(f"警告: {etf_code} 没有获取到数据")
                continue

            self.etf_data[etf_code] = data

        if not self.etf_data:
            raise ValueError("未能获取任何ETF数据")

        return self.etf_data

    def _initial_buy(self, date: pd.Timestamp):
        """
        初始建仓 - 使用开盘价买入

        Args:
            date: 建仓日期
        """
        # 基本打印信息
        print(f"\n在 {date.strftime('%Y-%m-%d')} 初始建仓")

        total_value = self.cash
        initial_transactions = {}

        for i, etf_code in enumerate(self.etf_codes):
            if etf_code not in self.etf_data:
                continue

            # 获取当天的开盘价
            if date not in self.etf_data[etf_code].index:
                # 如果当天没有数据，找最近的下一个交易日
                future_dates = self.etf_data[etf_code].index[self.etf_data[etf_code].index >= date]
                if len(future_dates) > 0:
                    date = future_dates[0]
                else:
                    if not self.verbose_trading:
                        print(f"  {etf_code}: 在 {date} 后没有交易数据，跳过")
                    continue

            open_price = self.etf_data[etf_code].loc[date, 'open']

            if pd.isna(open_price):
                if not self.verbose_trading:
                    print(f"  {etf_code}: 开盘价数据缺失，跳过")
                continue

            # 计算该ETF应分配的资金
            target_value = total_value * self.weights[i]

            # 计算可买入的股数（允许碎股）
            # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
            shares = target_value / (open_price * (1 + self.transaction_cost))

            if shares > 0:
                cost = shares * open_price * (1 + self.transaction_cost)

                # 更新持仓和现金
                self.positions[etf_code] = {
                    'shares': shares,
                    'avg_cost': open_price,
                    'total_cost': cost
                }
                self.cash -= cost

                # 记录交易
                transaction = {
                    'date': date,
                    'type': 'initial_buy',
                    'etf_code': etf_code,
                    'shares': shares,
                    'price': open_price,
                    'amount': cost,
                    'cash_after': self.cash
                }
                self.transactions.append(transaction)

                # 保存交易信息用于详细打印
                initial_transactions[etf_code] = {
                    'name': f'ETF {etf_code}',
                    'target_amount': target_value,
                    'shares': shares,
                    'price': open_price,
                    'actual_amount': cost
                }

                # 简单模式下只显示基本信息
                if not self.verbose_trading:
                    print(f"  买入 {etf_code}: {shares:.2f}股 @ ¥{open_price:.3f}, 成本: ¥{cost:,.2f}")

        # 根据参数决定打印详细程度
        if self.verbose_trading and initial_transactions:
            # 详细模式：调用详细信息打印方法
            self._print_initial_buy_details(initial_transactions)
        else:
            # 简单模式：只显示基本信息
            print(f"  建仓后现金: ¥{self.cash:,.2f}")

    def _get_dca_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """
        计算定投日期

        Args:
            trading_dates: 所有交易日列表

        Returns:
            List[pd.Timestamp]: 定投日期列表
        """
        if not self.enable_dca:
            return []

        dca_dates = []

        for date in trading_dates:
            # 跳过第一个交易日（这是初始建仓日）
            if date == trading_dates[0]:
                continue

            # 月度定投：每月最后一个交易日
            if self.dca_freq == 'monthly':
                # 检查是否是该月的最后一个交易日
                current_month = date.month
                current_year = date.year
                month_dates = [d for d in trading_dates if d.month == current_month and d.year == current_year]
                if date == month_dates[-1]:
                    dca_dates.append(date)

            # 年度定投：每年最后一个交易日
            elif self.dca_freq == 'yearly':
                # 检查是否是该年的最后一个交易日
                current_year = date.year
                year_dates = [d for d in trading_dates if d.year == current_year]
                if date == year_dates[-1]:
                    dca_dates.append(date)

        return dca_dates

    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """
        计算再平衡日期

        Args:
            trading_dates: 所有交易日列表

        Returns:
            List[pd.Timestamp]: 再平衡日期列表
        """
        if not self.enable_rebalancing:
            return []

        rebalance_dates = []

        for i, date in enumerate(trading_dates):
            # 跳过第一个交易日（这是初始建仓日）
            if i == 0:
                continue

            # 时间触发再平衡
            time_rebalance = False

            if self.rebalance_freq == 'monthly':
                # 每月最后一个交易日
                current_month = date.month
                current_year = date.year
                month_dates = [d for d in trading_dates if d.month == current_month and d.year == current_year]
                if date == month_dates[-1]:
                    time_rebalance = True

            elif self.rebalance_freq == 'quarterly':
                # 每季度最后一个交易日（3月、6月、9月、12月）
                if date.month in [3, 6, 9, 12]:
                    current_month = date.month
                    current_year = date.year
                    month_dates = [d for d in trading_dates if d.month == current_month and d.year == current_year]
                    if date == month_dates[-1]:
                        time_rebalance = True

            elif self.rebalance_freq == 'yearly':
                # 每年最后一个交易日
                current_year = date.year
                year_dates = [d for d in trading_dates if d.year == current_year]
                if date == year_dates[-1]:
                    time_rebalance = True

            # 阈值触发再平衡
            threshold_rebalance = self._check_rebalance_threshold(date)

            # 如果满足任一条件，则进行再平衡
            if time_rebalance or threshold_rebalance:
                rebalance_dates.append(date)

        return rebalance_dates

    def _check_rebalance_threshold(self, date: pd.Timestamp) -> bool:
        """
        检查是否需要基于阈值进行再平衡

        Args:
            date: 检查日期

        Returns:
            bool: 是否需要再平衡
        """
        if not self.positions:
            return False

        # 计算当前组合价值
        total_value = self._calculate_portfolio_value(date)
        if total_value <= 0:
            return False

        # 检查每个ETF的权重偏离
        for i, etf_code in enumerate(self.etf_codes):
            if etf_code in self.positions and etf_code in self.etf_data:
                if date in self.etf_data[etf_code].index:
                    current_price = self.etf_data[etf_code].loc[date, 'close']
                    current_value = self.positions[etf_code]['shares'] * current_price
                    current_weight = current_value / total_value
                    target_weight = self.weights[i]

                    # 检查权重偏离是否超过阈值
                    weight_deviation = abs(current_weight - target_weight)
                    if weight_deviation > self.rebalance_threshold:
                        return True

        return False

    def _dca_buy(self, date: pd.Timestamp):
        """
        定投买入 - 使用收盘价买入

        Args:
            date: 定投日期
        """
        # 基本打印信息
        print(f"\n定投 {date.strftime('%Y-%m-%d')}: 投入 ¥{self.dca_amount:,.2f}")

        # 添加定投资金
        self.cash += self.dca_amount
        
        # 记录现金流
        self.daily_flows[date] = self.daily_flows.get(date, 0) + self.dca_amount

        total_dca_value = self.dca_amount
        dca_transactions = []

        for i, etf_code in enumerate(self.etf_codes):
            if etf_code not in self.etf_data:
                continue

            # 获取当天的收盘价
            if date not in self.etf_data[etf_code].index:
                if not self.verbose_trading:
                    print(f"  {etf_code}: 当天无交易数据，跳过")
                continue

            close_price = self.etf_data[etf_code].loc[date, 'close']

            if pd.isna(close_price):
                if not self.verbose_trading:
                    print(f"  {etf_code}: 收盘价数据缺失，跳过")
                continue

            # 计算该ETF应分配的资金
            target_value = total_dca_value * self.weights[i]

            # 计算可买入的股数（允许碎股）
            # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
            shares = target_value / (close_price * (1 + self.transaction_cost))

            if shares > 0:
                cost = shares * close_price * (1 + self.transaction_cost)

                # 更新持仓和现金
                if etf_code in self.positions:
                    # 更新平均成本
                    old_shares = self.positions[etf_code]['shares']
                    old_cost = self.positions[etf_code]['total_cost']
                    new_shares = old_shares + shares
                    new_total_cost = old_cost + cost
                    avg_cost = new_total_cost / new_shares

                    self.positions[etf_code] = {
                        'shares': new_shares,
                        'avg_cost': avg_cost,
                        'total_cost': new_total_cost
                    }
                else:
                    # 新建持仓
                    self.positions[etf_code] = {
                        'shares': shares,
                        'avg_cost': close_price,
                        'total_cost': cost
                    }

                self.cash -= cost

                # 记录交易
                transaction = {
                    'date': date,
                    'type': 'dca_buy',
                    'etf_code': etf_code,
                    'shares': shares,
                    'price': close_price,
                    'amount': cost,
                    'cash_after': self.cash
                }
                self.transactions.append(transaction)
                dca_transactions.append(transaction)

                # 简单模式下只显示基本信息
                if not self.verbose_trading:
                    print(f"  买入 {etf_code}: {shares:.4f}股 @ ¥{close_price:.3f}, 成本: ¥{cost:,.2f}")

        # 根据参数决定打印详细程度
        if self.verbose_trading and dca_transactions:
            # 详细模式：调用详细信息打印方法
            self._print_dca_details(date, dca_transactions)
        else:
            # 简单模式：只显示基本信息
            print(f"  定投后现金: ¥{self.cash:,.2f}")

    def _print_initial_buy_details(self, transactions: Dict[str, Dict]):
        """打印初始建仓的详细信息

        Args:
            transactions: 交易记录字典，包含股票代码、交易数量、价格等信息
        """
        print("\n" + "=" * 80)
        print("初始建仓交易明细")
        print("=" * 80)

        # 打印每支股票的交易记录
        for stock_code, transaction in transactions.items():
            print(f"\n{transaction['name']} ({stock_code}):")
            print(f"  目标金额: ¥{transaction['target_amount']:,.2f}")
            print(f"  买入价格: ¥{transaction['price']:.4f}")
            print(f"  买入股数: {transaction['shares']:.2f}")
            print(f"  实际金额: ¥{transaction['actual_amount']:,.2f}")

        # 打印汇总信息
        total_actual = sum(t['actual_amount'] for t in transactions.values())
        remaining_cash = self.initial_capital - total_actual

        print(f"\n初始建仓汇总:")
        print(f"  初始资金: ¥{self.initial_capital:,.2f}")
        print(f"  总投入金额: ¥{total_actual:,.2f}")
        print(f"  剩余现金: ¥{remaining_cash:,.2f}")
        print(f"  建仓完成度: {(total_actual/self.initial_capital)*100:.1f}%")
        print("=" * 80)

    def _print_dca_details(self, date: pd.Timestamp, dca_transactions: List[Dict]):
        """
        打印定投交易的详细信息

        Args:
            date: 定投日期
            dca_transactions: 定投交易记录列表
        """
        print(f"\n定投 {date.strftime('%Y-%m-%d')}: 投入 ¥{self.dca_amount:,.2f}")
        print("  定投详情:")
        print("  " + "="*60)
        print(f"  {'ETF代码':<8} {'买入股数':<12} {'价格':<10} {'成本':<12}")
        print("  " + "-"*60)

        for transaction in dca_transactions:
            etf_code = transaction['etf_code']
            shares = transaction['shares']
            price = transaction['price']
            amount = transaction['amount']
            print(f"  {etf_code:<8} {shares:<12.4f} ¥{price:<9.3f} ¥{amount:<11,.2f}")

        print("  " + "="*60)
        print(f"  定投后现金: ¥{self.cash:,.2f}")

    def _rebalance_portfolio(self, date: pd.Timestamp):
        """
        执行组合再平衡 - 基于总资产价值进行完全再平衡

        Args:
            date: 再平衡日期
        """
        # 计算当前总资产价值（包括ETF和现金）
        total_assets = self._calculate_portfolio_value(date)
        if total_assets <= 0:
            if self.verbose_trading:
                print("  组合价值为0，跳过再平衡")
            return

        # 基本打印信息
        if not self.verbose_trading:
            print(f"\n再平衡 {date.strftime('%Y-%m-%d')}: 总资产价值: ¥{total_assets:,.0f}")
        else:
            print(f"\n再平衡 {date.strftime('%Y-%m-%d')}:")

        # 简单模式下的基本信息
        if self.verbose_trading:
            print(f"  总资产价值: ¥{total_assets:,.0f}")

        # 计算每个ETF的目标配置和当前配置的差额
        rebalance_plan = {}
        total_sell_needed = 0
        total_buy_needed = 0

        for i, etf_code in enumerate(self.etf_codes):
            if etf_code not in self.etf_data or date not in self.etf_data[etf_code].index:
                continue

            current_price = self.etf_data[etf_code].loc[date, 'close']
            target_value = total_assets * self.weights[i]

            # 获取当前持仓
            current_shares = 0
            current_value = 0
            if etf_code in self.positions:
                current_shares = self.positions[etf_code]['shares']
                current_value = current_shares * current_price

            # 计算需要调整的金额
            adjust_value = target_value - current_value

            rebalance_plan[etf_code] = {
                'current_shares': current_shares,
                'current_value': current_value,
                'target_value': target_value,
                'adjust_value': adjust_value,
                'price': current_price
            }

            if adjust_value > 0:
                total_buy_needed += adjust_value
            else:
                total_sell_needed += abs(adjust_value)

        # 执行再平衡：先卖后买
        total_sell_proceeds = 0
        total_buy_cost = 0
        has_trades = False

        # 第一阶段：执行所有卖出操作
        for etf_code, plan in rebalance_plan.items():
            if plan['adjust_value'] < 0:  # 需要卖出
                shares_to_sell = abs(plan['adjust_value']) / plan['price']
                shares_to_sell = min(shares_to_sell, plan['current_shares'])  # 不能超过持有数量

                if shares_to_sell > 0:
                    has_trades = True
                    sell_proceeds = shares_to_sell * plan['price'] * (1 - self.transaction_cost)
                    remaining_shares = plan['current_shares'] - shares_to_sell

                    # 更新持仓
                    if remaining_shares > 0:
                        remaining_cost_ratio = remaining_shares / plan['current_shares']
                        new_total_cost = self.positions[etf_code]['total_cost'] * remaining_cost_ratio
                        avg_cost = new_total_cost / remaining_shares

                        self.positions[etf_code] = {
                            'shares': remaining_shares,
                            'avg_cost': avg_cost,
                            'total_cost': new_total_cost
                        }
                    else:
                        # 全部卖出
                        del self.positions[etf_code]

                    self.cash += sell_proceeds
                    total_sell_proceeds += sell_proceeds

                    # 记录交易
                    self.transactions.append({
                        'date': date,
                        'type': 'rebalance_sell',
                        'etf_code': etf_code,
                        'shares': shares_to_sell,
                        'price': plan['price'],
                        'amount': sell_proceeds,
                        'cash_after': self.cash
                    })

                    # 详细模式下显示基本交易信息
                    if self.verbose_trading:
                        print(f"  卖出 {etf_code}: {shares_to_sell:.2f}股 @ ¥{plan['price']:.3f}, 收入: ¥{sell_proceeds:,.0f}")

        # 第二阶段：执行所有买入操作
        available_cash = self.cash  # 包括原有现金和卖出所得

        for etf_code, plan in rebalance_plan.items():
            if plan['adjust_value'] > 0:  # 需要买入
                # 计算实际可买入金额（考虑现金限制）
                max_affordable_value = available_cash / (1 + self.transaction_cost)
                actual_buy_value = min(plan['adjust_value'], max_affordable_value)

                if actual_buy_value > 0:
                    has_trades = True
                    shares_to_buy = actual_buy_value / (plan['price'] * (1 + self.transaction_cost))
                    cost = shares_to_buy * plan['price'] * (1 + self.transaction_cost)

                    # 更新或新建持仓
                    if etf_code in self.positions:
                        new_shares = self.positions[etf_code]['shares'] + shares_to_buy
                        new_total_cost = self.positions[etf_code]['total_cost'] + cost
                        avg_cost = new_total_cost / new_shares

                        self.positions[etf_code] = {
                            'shares': new_shares,
                            'avg_cost': avg_cost,
                            'total_cost': new_total_cost
                        }
                    else:
                        self.positions[etf_code] = {
                            'shares': shares_to_buy,
                            'avg_cost': plan['price'],
                            'total_cost': cost
                        }

                    self.cash -= cost
                    available_cash -= cost
                    total_buy_cost += cost

                    # 记录交易
                    self.transactions.append({
                        'date': date,
                        'type': 'rebalance_buy',
                        'etf_code': etf_code,
                        'shares': shares_to_buy,
                        'price': plan['price'],
                        'amount': cost,
                        'cash_after': self.cash
                    })

                    # 详细模式下显示基本交易信息
                    if self.verbose_trading:
                        print(f"  买入 {etf_code}: {shares_to_buy:.2f}股 @ ¥{plan['price']:.3f}, 成本: ¥{cost:,.0f}")

        # 根据参数决定打印详细程度
        if self.verbose_trading and has_trades:
            # 详细模式：调用详细信息打印方法
            self._print_rebalance_details(date, total_assets, rebalance_plan,
                                         total_buy_needed, total_sell_needed,
                                         total_sell_proceeds, total_buy_cost)
        elif has_trades:
            # 简单模式：只显示汇总信息
            print(f"  交易汇总: 卖出¥{total_sell_proceeds:,.0f}, 买入¥{total_buy_cost:,.0f}, 再平衡后: ¥{self._calculate_portfolio_value(date):,.0f}")
        elif not self.verbose_trading:
            print("  无需调整")

    def _print_pre_rebalance_analysis(self, date: pd.Timestamp, total_value: float):
        """
        打印平衡前的ETF持仓详情（简洁版）

        Args:
            date: 再平衡日期
            total_value: 当前组合总价值
        """
        print(f"\n  ETF持仓分析:")
        print("  " + "="*70)
        print(f"  {'ETF代码':<8} {'股数':<10} {'市值':<12} {'当前权重':<10} {'目标权重':<10} {'偏离':<8}")
        print("  " + "-"*70)

        for i, etf_code in enumerate(self.etf_codes):
            # 获取当前持仓信息
            shares = 0
            if etf_code in self.positions:
                shares = self.positions[etf_code]['shares']

            # 获取当前价格
            price = None
            if etf_code in self.etf_data and date in self.etf_data[etf_code].index:
                price = self.etf_data[etf_code].loc[date, 'close']

            market_value = shares * price if price is not None else 0
            weight = (market_value / total_value * 100) if total_value > 0 else 0
            target_weight = self.weights[i] * 100
            weight_diff = weight - target_weight

            print(f"  {etf_code:<8} {shares:<10.2f} ¥{market_value:<11,.0f} {weight:<9.1f}% {target_weight:<9.1f}% {weight_diff:+7.1f}%")

        print("  " + "="*70)
        print(f"  组合价值: ¥{total_value:,.0f}, 现金: ¥{self.cash:,.0f}")
        print()

    def _print_rebalance_calculation_details(self, rebalance_plan: Dict, total_buy_needed: float, total_sell_needed: float):
        """
        打印再平衡计算的详细过程和交易计划

        Args:
            rebalance_plan: 再平衡计划字典
            total_buy_needed: 总需要买入金额
            total_sell_needed: 总需要卖出金额
        """
        print(f"\n  再平衡计算详情:")
        print("  " + "="*80)
        print(f"  {'ETF代码':<8} {'当前市值':<12} {'目标市值':<12} {'调整金额':<12} {'调整股数':<10} {'操作类型':<8}")
        print("  " + "-"*80)

        for etf_code, plan in rebalance_plan.items():
            current_value = plan['current_value']
            target_value = plan['target_value']
            adjust_value = plan['adjust_value']
            price = plan['price']

            if adjust_value > 0:
                # 需要买入
                shares_needed = adjust_value / (price * (1 + self.transaction_cost))
                action = "买入"
                print(f"  {etf_code:<8} ¥{current_value:<11,.0f} ¥{target_value:<11,.0f} +¥{adjust_value:<11,.0f} {shares_needed:<9.2f} {action:<8}")
            elif adjust_value < 0:
                # 需要卖出
                shares_to_sell = abs(adjust_value) / price
                action = "卖出"
                print(f"  {etf_code:<8} ¥{current_value:<11,.0f} ¥{target_value:<11,.0f} -¥{abs(adjust_value):<11,.0f} {shares_to_sell:<9.2f} {action:<8}")
            else:
                # 无需调整
                action = "无操作"
                print(f"  {etf_code:<8} ¥{current_value:<11,.0f} ¥{target_value:<11,.0f} ¥{adjust_value:<11,.0f} {'0.00':<9} {action:<8}")

        print("  " + "="*80)
        print(f"  交易汇总: 预计卖出 ¥{total_sell_needed:,.0f}, 预计买入 ¥{total_buy_needed:,.0f}")
        print(f"  净调整: {'¥' if total_buy_needed > total_sell_needed else '-¥'}{abs(total_buy_needed - total_sell_needed):,.0f}")
        print()

    def _print_post_rebalance_summary(self, date: pd.Timestamp, total_value: float):
        """
        打印平衡后的ETF持仓摘要（简洁版）

        Args:
            date: 再平衡日期
            total_value: 当前组合总价值
        """
        # 计算再平衡后的总价值
        after_value = self._calculate_portfolio_value(date)

        print(f"\n  再平衡后: 组合价值¥{after_value:,.0f}, 现金¥{self.cash:,.0f}")

    def _print_rebalance_details(self, date: pd.Timestamp, total_assets: float, rebalance_plan: Dict,
                                 total_buy_needed: float, total_sell_needed: float,
                                 total_sell_proceeds: float, total_buy_cost: float):
        """
        打印再平衡的详细信息

        Args:
            date: 再平衡日期
            total_assets: 当前总资产价值
            rebalance_plan: 再平衡计划字典
            total_buy_needed: 总需要买入金额
            total_sell_needed: 总需要卖出金额
            total_sell_proceeds: 实际卖出所得
            total_buy_cost: 实际买入成本
        """
        print(f"\n再平衡 {date.strftime('%Y-%m-%d')}:")
        print(f"  总资产价值: ¥{total_assets:,.0f}")

        # 显示平衡前的ETF持仓详情
        self._print_pre_rebalance_analysis(date, total_assets)

        # 显示再平衡计算详情和交易计划
        self._print_rebalance_calculation_details(rebalance_plan, total_buy_needed, total_sell_needed)

        # 显示实际交易执行情况
        print(f"  实际交易执行:")
        print("  " + "="*60)
        print(f"  {'ETF代码':<8} {'交易类型':<8} {'股数':<12} {'价格':<10} {'金额':<12}")
        print("  " + "-"*60)

        # 从交易记录中提取当天的再平衡交易
        rebalance_transactions = [t for t in self.transactions if t['date'] == date and t['type'].startswith('rebalance')]

        for transaction in rebalance_transactions:
            etf_code = transaction['etf_code']
            transaction_type = '卖出' if transaction['type'] == 'rebalance_sell' else '买入'
            shares = transaction['shares']
            price = transaction['price']
            amount = transaction['amount']
            print(f"  {etf_code:<8} {transaction_type:<8} {shares:<12.2f} ¥{price:<9.3f} ¥{amount:<11,.0f}")

        print("  " + "="*60)
        print(f"  交易汇总: 卖出¥{total_sell_proceeds:,.0f}, 买入¥{total_buy_cost:,.0f}")
        print(f"  剩余现金: ¥{self.cash:,.0f}")

        # 显示平衡后的ETF持仓详情
        final_value = self._calculate_portfolio_value(date)
        self._print_post_rebalance_summary(date, final_value)

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """
        计算指定日期的组合总价值

        Args:
            date: 计算价值的日期

        Returns:
            float: 组合总价值
        """
        total_value = self.cash

        for etf_code, position in self.positions.items():
            if etf_code not in self.etf_data:
                continue

            # 获取当天的收盘价
            if date in self.etf_data[etf_code].index:
                close_price = self.etf_data[etf_code].loc[date, 'close']
                if not pd.isna(close_price):
                    total_value += position['shares'] * close_price
            else:
                # 如果当天没有数据，使用前一个有效交易日的价格（停盘处理）
                etf_data = self.etf_data[etf_code]
                prior_dates = etf_data.index[etf_data.index < date]
                if len(prior_dates) > 0:
                    last_valid_date = prior_dates[-1]
                    close_price = etf_data.loc[last_valid_date, 'close']
                    if not pd.isna(close_price):
                        total_value += position['shares'] * close_price

        return total_value

    def run_backtest(self):
        """
        运行回测 - 专注于计算逻辑，减少打印输出

        执行顺序：
        1. 初始建仓（第一个交易日）
        2. 逐日处理：
           - 先执行再平衡（如果当天是再平衡日）
           - 后执行定投（如果当天是定投日）
        3. 计算每日组合价值

        优化说明：
        - 采用"先平衡后定投"策略，避免定投买入后立即卖出
        - 减少逻辑冲突和交易成本
        - 更符合实际投资操作逻辑
        """
        # 获取数据
        self.fetch_data()

        # 获取所有交易日
        all_dates = set()
        for data in self.etf_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(all_dates)

        # 过滤回测期间的交易日
        trading_dates = [date for date in trading_dates if date >= self.start_date and date <= self.end_date]

        # 计算定投日期
        self.dca_dates = self._get_dca_dates(trading_dates)

        # 计算再平衡日期
        self.rebalance_dates = self._get_rebalance_dates(trading_dates)

        # 第一天初始建仓
        if trading_dates:
            self._initial_buy(trading_dates[0])

        # 逐日更新 - 采用先平衡后定投策略
        for i, date in enumerate(trading_dates):
            # 先执行再平衡：调整现有组合到目标权重
            if date in self.rebalance_dates:
                self._rebalance_portfolio(date)

            # 后执行定投：将新资金按目标权重投入
            if date in self.dca_dates:
                self._dca_buy(date)

            # 计算当日组合价值
            portfolio_value = self._calculate_portfolio_value(date)

            # 记录每日持仓详情
            daily_position_detail = {}
            total_value = portfolio_value

            for i, etf_code in enumerate(self.etf_codes):
                if etf_code in self.positions and etf_code in self.etf_data:
                    close_price = None

                    if date in self.etf_data[etf_code].index:
                        close_price = self.etf_data[etf_code].loc[date, 'close']
                    else:
                        # 如果当天没有数据，使用前一个有效交易日的价格（停盘处理）
                        etf_data = self.etf_data[etf_code]
                        prior_dates = etf_data.index[etf_data.index < date]
                        if len(prior_dates) > 0:
                            last_valid_date = prior_dates[-1]
                            close_price = etf_data.loc[last_valid_date, 'close']

                    if close_price is not None and not pd.isna(close_price):
                        shares = self.positions[etf_code]['shares']
                        value = shares * close_price
                        weight = value / total_value if total_value > 0 else 0

                        daily_position_detail[etf_code] = {
                            'shares': shares,
                            'value': value,
                            'weight': weight,
                            'price': close_price
                        }

            # 添加现金信息
            daily_position_detail['cash'] = {
                'value': self.cash,
                'weight': self.cash / total_value if total_value > 0 else 0
            }

            # 记录
            self.daily_dates.append(date)
            self.daily_values.append(portfolio_value)
            self.daily_positions[date] = daily_position_detail

        # 计算回测结果
        self._calculate_results()

    def _calculate_daily_cumulative_investment(self):
        """
        计算每日的累计投入金额

        Returns:
            List[float]: 每日累计投入金额列表，与daily_dates对应
        """
        if not self.daily_dates:
            return []

        # 将定投日期转换为集合以便快速查找
        dca_dates_set = set(self.dca_dates)

        daily_investments = []
        cumulative_investment = self.initial_capital

        for date in self.daily_dates:
            # 如果这天是定投日，增加累计投入
            if date in dca_dates_set:
                cumulative_investment += self.dca_amount

            daily_investments.append(cumulative_investment)

        return daily_investments

    def _get_investment_at_dates(self, dates):
        """
        获取指定日期的累计投入金额

        Args:
            dates: 日期列表或单个日期

        Returns:
            List[float] or float: 对应日期的累计投入金额
        """
        if not self.daily_dates:
            return [0] * len(dates) if isinstance(dates, list) else 0

        daily_investments = self._calculate_daily_cumulative_investment()
        date_to_investment = dict(zip(self.daily_dates, daily_investments))

        if isinstance(dates, list):
            return [date_to_investment.get(date, 0) for date in dates]
        else:
            return date_to_investment.get(dates, 0)

    def _calculate_period_return(self, start_date, end_date, start_value, end_value, use_period_investment=True):
        """
        计算特定时间段的收益率

        Args:
            start_date: 开始日期
            end_date: 结束日期
            start_value: 开始时资产价值
            end_value: 结束时资产价值
            use_period_investment: 是否使用期间累计投入计算收益率
                                 True - 累计收益率（用于折线图）
                                 False - 独立期间收益率（用于热力图和柱状图）

        Returns:
            float: 收益率百分比
        """

        # 获取开始和结束时的累计投入
        start_investment = self._get_investment_at_dates(start_date)
        end_investment = self._get_investment_at_dates(end_date)



        if use_period_investment:
            # 计算累计收益率（考虑期间所有投入）
            net_investment = end_investment - start_investment
            if net_investment > 0:
                # 有定投，使用加权平均计算
                total_invested = start_investment + net_investment
                return (end_value / total_invested - 1) * 100
            else:
                # 没有定投，使用简单计算
                if start_investment > 0:
                    return (end_value / start_investment - 1) * 100
                else:
                    return 0
        else:
            # 计算独立期间收益率（考虑期初投入，排除期间定投对期末价值的影响）
            # 这适用于热力图和年度柱状图，显示每个时期的独立收益率
            net_investment = end_investment - start_investment
            
            # 如果有定投，从期末价值中减去定投金额
            # 这样计算的是：(期末价值 - 本期投入) / 期初价值 - 1
            adjusted_end_value = end_value
            if net_investment > 0:
                adjusted_end_value = end_value - net_investment
            
            if start_value > 0:

                return (adjusted_end_value / start_value - 1) * 100
            else:
                return 0

    def _calculate_results(self):
        """计算回测结果指标"""
        if not self.daily_values:
            return

        # 转换为Series便于计算
        value_series = pd.Series(self.daily_values, index=self.daily_dates)

        # 计算总投入（包括定投）
        total_investment = self.initial_capital
        if self.enable_dca:
            total_investment += self.dca_amount * len(self.dca_dates)

        # --- 修正收益率计算逻辑 (TWR) ---
        # 1. 计算每日调整后收益率 (剔除现金流影响)
        adjusted_returns = []
        
        # 第一天的收益率：(第一天价值 - 第一天现金流) / 初始资金 - 1
        # 注意：初始建仓日通常没有额外现金流，除非第一天就定投
        first_date = self.daily_dates[0]
        first_flow = self.daily_flows.get(first_date, 0)
        # 初始资金视为t=0时的价值
        r0 = (self.daily_values[0] - first_flow) / self.initial_capital - 1
        adjusted_returns.append(r0)
        
        for i in range(1, len(self.daily_values)):
            date = self.daily_dates[i]
            current_value = self.daily_values[i]
            prev_value = self.daily_values[i-1]
            flow = self.daily_flows.get(date, 0)
            
            # 修正后的收益率公式: (Vt - Ct) / Vt-1 - 1
            if prev_value > 0:
                r_t = (current_value - flow) / prev_value - 1
            else:
                r_t = 0
            adjusted_returns.append(r_t)
            
        returns_series = pd.Series(adjusted_returns, index=self.daily_dates)
        
        # 2. 计算时间加权收益率 (TWR)
        # TWR = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        twr_cumulative = (1 + returns_series).cumprod()
        total_twr = twr_cumulative.iloc[-1] - 1
        
        # 3. 计算投资回报率 (ROI) - 简单的总收益率
        # ROI = (最终价值 / 总投入) - 1
        roi = (value_series.iloc[-1] / total_investment - 1) * 100

        # 年化收益率 (基于TWR)
        days = (value_series.index[-1] - value_series.index[0]).days
        years = days / 365.25
        if years > 0:
            # 使用TWR计算年化
            annual_return = (1 + total_twr) ** (1/years) - 1
            annual_return_pct = annual_return * 100
        else:
            annual_return_pct = 0

        # 最大回撤 (基于实际账户价值，这是投资者真实的体验)
        rolling_max = value_series.expanding().max()
        drawdown = (value_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # 计算最大回撤回补周期
        max_dd_idx = drawdown.idxmin()  # 最大回撤发生的日期
        max_dd_value = drawdown.min()    # 最大回撤深度

        # 找到对应的滚动最高点（前期高点）
        pre_peak_value = rolling_max.loc[max_dd_idx]

        # 从最大回撤点开始，找到首次超过前期高点的日期
        recovery_idx = None
        recovery_date = None

        # 检查最大回撤之后的所有日期
        after_max_dd = value_series.loc[max_dd_idx:]

        for date, value in after_max_dd.items():
            if value >= pre_peak_value:
                recovery_idx = date
                recovery_date = date
                break

        # 计算回补周期
        if recovery_idx is not None:
            # 交易日数
            max_drawdown_recovery_days = (recovery_idx - max_dd_idx).days
            # 日历天数（更直观的显示）
            max_drawdown_recovery_calendar_days = (recovery_idx - max_dd_idx).days
        else:
            # 如果到回测结束都未回补
            max_drawdown_recovery_days = None
            max_drawdown_recovery_calendar_days = None
            recovery_idx = None

        # 年化波动率 (基于调整后的每日收益率)
        volatility = returns_series.std() * np.sqrt(252)

        # 夏普比率计算 (基于调整后的年化收益率和波动率)
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        self.results = {
            'initial_capital': self.initial_capital,
            'total_investment': total_investment,
            'final_value': value_series.iloc[-1],
            'total_return': roi,  # 使用ROI作为总收益率显示
            'annual_return': annual_return_pct, # 使用TWR年化
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_dd_idx,
            'pre_peak_value': pre_peak_value,
            'recovery_date': recovery_idx,
            'max_drawdown_recovery_days': max_drawdown_recovery_days,
            'max_drawdown_recovery_calendar_days': max_drawdown_recovery_calendar_days,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'risk_free_rate': self.risk_free_rate,
            'trading_days': len(self.daily_values),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'dca_count': len(self.dca_dates) if self.enable_dca else 0,
            'rebalance_count': len(self.rebalance_dates) if self.enable_rebalancing else 0
        }

    def print_results(self):
        """打印回测结果"""
        print("\n" + "="*50)
        print("回测结果")
        print("="*50)

        # 显示投入情况
        print(f"初始资金: ¥{self.results['initial_capital']:,.2f}")
        if self.enable_dca:
            print(f"定投总额: ¥{self.dca_amount * self.results['dca_count']:,.2f} ({self.results['dca_count']}次)")
        if self.enable_rebalancing:
            print(f"再平衡次数: {self.results['rebalance_count']}次")
        print(f"总投入: ¥{self.results['total_investment']:,.2f}")

        print(f"最终价值: ¥{self.results['final_value']:,.2f}")
        print(f"总收益率: {self.results['total_return']:.2f}%")
        print(f"年化收益率: {self.results['annual_return']:.2f}%")
        print(f"最大回撤: {self.results['max_drawdown']:.2f}%")

        # 显示最大回撤回补周期
        if self.results['max_drawdown_recovery_days'] is not None:
            print(f"最大回撤回补周期: {self.results['max_drawdown_recovery_days']}个交易日 "
                  f"({self.results['max_drawdown_recovery_calendar_days']}天日历时间)")
            print(f"最大回撤发生: {self.results['max_drawdown_date'].strftime('%Y-%m-%d')}")
            print(f"回补完成: {self.results['recovery_date'].strftime('%Y-%m-%d')}")
        else:
            print(f"最大回撤回补周期: 未完全回补 (截至回测结束)")
            print(f"最大回撤发生: {self.results['max_drawdown_date'].strftime('%Y-%m-%d')}")

        print(f"年化波动率: {self.results['volatility']:.2f}%")
        print(f"夏普比率: {self.results['sharpe_ratio']:.3f} (无风险利率: {self.results['risk_free_rate']:.1%})")
        print(f"交易天数: {self.results['trading_days']}")
        print(f"回测期间: {self.results['start_date'].strftime('%Y-%m-%d')} 至 {self.results['end_date'].strftime('%Y-%m-%d')}")

        # 打印最终持仓
        print("\n最终持仓:")
        print("-" * 50)
        total_value = self.results['final_value']
        for etf_code, position in self.positions.items():
            last_date = self.daily_dates[-1]
            if etf_code in self.etf_data and last_date in self.etf_data[etf_code].index:
                current_price = self.etf_data[etf_code].loc[last_date, 'close']
                market_value = position['shares'] * current_price
                percentage = market_value / total_value * 100
                profit_loss = market_value - position['total_cost']
                profit_pct = (profit_loss / position['total_cost'] * 100) if position['total_cost'] > 0 else 0
                print(f"{etf_code}: {position['shares']:.2f}股, 市值: ¥{market_value:,.2f} ({percentage:.1f}%)")
                print(f"  成本: ¥{position['total_cost']:,.2f}, 盈亏: ¥{profit_loss:,.2f} ({profit_pct:+.2f}%)")

        print(f"现金: ¥{self.cash:,.2f} ({self.cash/total_value*100:.1f}%)")

    def _get_rebalance_period_start(self, current_rebalance_date: pd.Timestamp) -> pd.Timestamp:
        """
        获取当前再平衡周期的开始日期（上一个再平衡日之后的第一天）

        Args:
            current_rebalance_date: 当前再平衡日期

        Returns:
            pd.Timestamp: 当前再平衡周期的开始日期
        """
        # 获取所有交易日
        all_dates = set()
        for data in self.etf_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(all_dates)

        # 过滤回测期间的交易日
        trading_dates = [date for date in trading_dates if date >= self.start_date and date <= self.end_date]

        # 找到当前再平衡日期在交易日列表中的索引
        try:
            current_index = trading_dates.index(current_rebalance_date)
        except ValueError:
            # 如果找不到，返回回测开始日期
            return self.start_date

        # 如果是第一个再平衡日，返回回测开始日期
        if current_index == 0:
            return self.start_date

        # 找到上一个再平衡日期
        previous_rebalance_date = None
        for i in range(current_index - 1, -1, -1):
            if trading_dates[i] in self.rebalance_dates:
                previous_rebalance_date = trading_dates[i]
                break

        # 如果没有找到上一个再平衡日期（比如第一个再平衡日），返回回测开始日期
        if previous_rebalance_date is None:
            return self.start_date

        # 返回上一个再平衡日之后的第一个交易日
        for i in range(current_index):
            if trading_dates[i] > previous_rebalance_date:
                return trading_dates[i]

        # 如果没找到，返回回测开始日期
        return self.start_date

    def _print_rebalancing_details(self, period_start: pd.Timestamp, rebalance_date: pd.Timestamp, total_value: float):
        """
        打印每个ETF的再平衡信息，单独一行显示

        Args:
            period_start: 再平衡周期开始日期
            rebalance_date: 再平衡执行日期
            total_value: 当前组合总价值
        """
        start_total_value = self._calculate_portfolio_value(period_start)
        before_total_value = self._calculate_portfolio_value(rebalance_date)

        if start_total_value > 0 and before_total_value > 0:
            for i, etf_code in enumerate(self.etf_codes):
                # 周期开始时的价值
                start_value = 0
                if etf_code in self.positions and etf_code in self.etf_data:
                    if period_start in self.etf_data[etf_code].index:
                        close_price = self.etf_data[etf_code].loc[period_start, 'close']
                        start_value = self.positions[etf_code]['shares'] * close_price

                # 周期结束时的价值
                before_value = 0
                if etf_code in self.positions and etf_code in self.etf_data:
                    if rebalance_date in self.etf_data[etf_code].index:
                        close_price = self.etf_data[etf_code].loc[rebalance_date, 'close']
                        before_value = self.positions[etf_code]['shares'] * close_price

                # 目标价值
                target_value = total_value * self.weights[i]

                # 计算权重
                start_weight = (start_value / start_total_value * 100) if start_total_value > 0 else 0
                before_weight = (before_value / before_total_value * 100) if before_total_value > 0 else 0
                target_weight = self.weights[i] * 100

                # 计算偏离
                deviation = before_weight - target_weight

                # 预测需要进行的交易
                adjust_value = target_value - before_value
                if abs(adjust_value) < 0.01:  # 很小的偏离，无操作
                    trade_info = f"偏离 {deviation:+.1f}%，无操作"
                elif adjust_value > 0:
                    # 需要买入
                    trade_info = f"偏离 {deviation:+.1f}%，预计买入 ¥{adjust_value:,.0f}"
                else:
                    # 需要卖出
                    trade_info = f"偏离 {deviation:+.1f}%，预计卖出 ¥{abs(adjust_value):,.0f}"

                # 打印该ETF的信息
                print(f"  {etf_code}: ¥{start_value:,.0f}({start_weight:.1f}%) → ¥{before_value:,.0f}({before_weight:.1f}%) → ¥{target_value:,.0f}({target_weight:.0f}%)   {trade_info}")

    def generate_report(self, show_plot=True):
        """
        生成完整的可视化报告 - 整合图表显示和汇总信息输出

        Args:
            show_plot: 是否显示图表
        """
        # 显示回测结果汇总
        self.print_results()

        if not self.daily_values or not self.daily_dates:
            print("没有数据可用于生成图表")
            return

        if show_plot:
            self._plot_combined_dashboard()


    def _plot_combined_dashboard(self):
        """生成包含5个子图的合并仪表板"""
        # 创建5x1的子图布局，前两个图独占一排，后面三个图共享一排
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=('账户资金变化', '累计收益率', '回撤分析', '月度收益率热力图', '年度收益率'),
            specs=[
                [{"secondary_y": False}],  # 账户资金变化
                [{"secondary_y": False}],  # 累计收益率
                [{"secondary_y": False}],  # 回撤分析
                [{"type": "heatmap"}],    # 月度收益率热力图
                [{"type": "bar"}]         # 年度收益率
            ],
            vertical_spacing=0.08
        )

        # 1. 账户资金变化折线图 (第1行)
        self._add_portfolio_value_subplot(fig, row=1, col=1)

        # 2. 收益率折线图 (第2行)
        self._add_returns_subplot(fig, row=2, col=1)

        # 3. 回撤图 (第3行)
        self._add_drawdown_subplot(fig, row=3, col=1)

        # 4. 月度收益率热力图 (第4行)
        self._add_monthly_heatmap_subplot(fig, row=4, col=1)

        # 5. 年度收益率柱状图 (第5行)
        self._add_annual_returns_subplot(fig, row=5, col=1)

        # 更新整体布局
        fig.update_layout(
            title={
                'text': f'投资组合回测报告 ({self.results["start_date"].strftime("%Y-%m-%d")} 至 {self.results["end_date"].strftime("%Y-%m-%d")})',
                'x': 0.5,
                'font': {'size': 24}
            },
            template='plotly_white',
            showlegend=True,
            height=2400,  # 增加高度以容纳更多行
            width=1000,  # 增加宽度以拉长x轴
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"portfolio_backtest_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def _add_portfolio_value_subplot(self, fig, row, col):
        """添加账户资金变化子图"""
        # 创建时间序列数据，确保正确的DatetimeIndex
        dates = pd.to_datetime(self.daily_dates)

        # 主线：总资产价值
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.daily_values,
            mode='lines',
            name='总资产价值',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                         '总资产: ¥%{y:,.0f}<extra></extra>'
        ), row=row, col=col)

        # 添加再平衡标记
        if self.rebalance_dates:
            rebalance_values = []
            for date in self.rebalance_dates:
                # 找到最接近的日期索引
                date_timestamp = pd.to_datetime(date)
                closest_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - date_timestamp))
                rebalance_values.append(self.daily_values[closest_idx])

            fig.add_trace(go.Scatter(
                x=pd.to_datetime(self.rebalance_dates),
                y=rebalance_values,
                mode='markers',
                name='再平衡',
                marker=dict(color='red', size=6, symbol='triangle-down'),
                hovertemplate='<b>再平衡</b><br>日期: %{x|%Y-%m-%d}<br>资产: ¥%{y:,.0f}<extra></extra>',
                showlegend=False
            ), row=row, col=col)

        # 计算Y轴范围（基于实际数据，上下浮动10%）
        min_value = min(self.daily_values)
        max_value = max(self.daily_values)
        y_range = max_value - min_value
        y_padding = y_range * 0.1
        y_min = min_value - y_padding
        y_max = max_value + y_padding

        # 更新子图布局
        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(
            title_text="账户价值 (¥)",
            tickformat=',.0f',
            range=[y_min, y_max],
            row=row,
            col=col
        )

        # 确保显示完整的时间范围
        fig.update_xaxes(range=[dates.min(), dates.max()], row=row, col=col)

    def _add_returns_subplot(self, fig, row, col):
        """添加收益率子图"""
        # 计算每日累计投入
        daily_investments = self._calculate_daily_cumulative_investment()

        # 使用每日累计投入计算收益率
        returns_data = []
        for value, daily_investment in zip(self.daily_values, daily_investments):
            if daily_investment > 0:
                return_pct = (value / daily_investment - 1) * 100
            else:
                return_pct = 0
            returns_data.append(return_pct)

        dates = pd.to_datetime(self.daily_dates)

        # 主线：累计收益率
        fig.add_trace(go.Scatter(
            x=dates,
            y=returns_data,
            mode='lines',
            name='累计收益率',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                         '收益率: %{y:.2f}%<extra></extra>',
            showlegend=False
        ), row=row, col=col)

        # 添加0%基准线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

        # 更新子图布局
        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="收益率 (%)", tickformat='.1f', row=row, col=col)

    def _add_drawdown_subplot(self, fig, row, col):
        """添加回撤子图"""
        # 创建时间序列
        dates = pd.to_datetime(self.daily_dates)
        value_series = pd.Series(self.daily_values, index=dates)
        rolling_max = value_series.expanding().max()
        drawdown = (value_series - rolling_max) / rolling_max * 100

        # 回撤线
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='回撤',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                         '回撤: %{y:.2f}%<extra></extra>',
            showlegend=False
        ), row=row, col=col)

        # 标记最大回撤点
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()

        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers',
            name='最大回撤',
            marker=dict(color='darkred', size=10, symbol='x'),
            hovertemplate='<b>最大回撤</b><br>' +
                         '日期: %{x|%Y-%m-%d}<br>' +
                         '回撤: %{y:.2f}%<extra></extra>',
            showlegend=False
        ), row=row, col=col)

        # 更新子图布局
        fig.update_xaxes(title_text="日期", row=row, col=col)
        fig.update_yaxes(title_text="回撤 (%)", tickformat='.1f', row=row, col=col)

    def _add_monthly_heatmap_subplot(self, fig, row, col):
        """添加月度收益率热力图子图"""
        # 准备月度数据
        df = pd.DataFrame({
            'date': self.daily_dates,
            'value': self.daily_values
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # 计算月度收益率
        monthly_data = []
        for (year, month), group in df.groupby(['year', 'month']):
            if len(group) > 1:
                start_date = group.iloc[0]['date']
                end_date = group.iloc[-1]['date']
                start_value = group.iloc[0]['value']
                end_value = group.iloc[-1]['value']

                # 计算月度独立收益率
                month_return = self._calculate_period_return(start_date, end_date, start_value, end_value, use_period_investment=False)

                monthly_data.append({
                    'year': year,
                    'month': month,
                    'return': month_return
                })

        if not monthly_data:
            return

        # 创建数据框
        heatmap_df = pd.DataFrame(monthly_data)
        heatmap_pivot = heatmap_df.pivot(index='year', columns='month', values='return')

        # 确保年份是整数类型
        years = heatmap_pivot.index.astype(int)

        # 添加热力图
        fig.add_trace(go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=years,
            colorscale='RdYlGn',
            text=heatmap_pivot.round(2).values,
            texttemplate='%{text}%',
            textfont=dict(size=8),
            hoverongaps=False,
            colorbar=dict(
                title="收益率 (%)",
                title_side="right",
                len=0.4
            ),
            hovertemplate='年份: %{y}<br>月份: %{x}<br>收益率: %{z:.2f}%<extra></extra>',
            showscale=False
        ), row=row, col=col)

        # 更新子图布局
        fig.update_xaxes(title_text="月份", row=row, col=col)
        fig.update_yaxes(
            title_text="年份",
            autorange="reversed",
            tickmode='array',
            tickvals=years,
            ticktext=years,
            row=row,
            col=col
        )

    def _add_annual_returns_subplot(self, fig, row, col):
        """添加年度收益率柱状图子图"""
        # 准备年度数据
        df = pd.DataFrame({
            'date': self.daily_dates,
            'value': self.daily_values
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        # 计算年度收益率
        annual_data = []
        for year, group in df.groupby('year'):
            if len(group) > 1:
                start_date = group.iloc[0]['date']
                end_date = group.iloc[-1]['date']
                start_value = group.iloc[0]['value']
                end_value = group.iloc[-1]['value']

                # 计算年度独立收益率
                year_return = self._calculate_period_return(start_date, end_date, start_value, end_value, use_period_investment=False)

                annual_data.append({
                    'year': year,
                    'return': year_return
                })

        if not annual_data:
            return

        # 创建数据框
        annual_df = pd.DataFrame(annual_data)

        # 确定颜色（正收益绿色，负收益红色）
        colors = ['green' if ret >= 0 else 'red' for ret in annual_df['return']]

        # 添加柱状图
        fig.add_trace(go.Bar(
            x=annual_df['year'],
            y=annual_df['return'],
            name='年度收益率',
            marker_color=colors,
            text=annual_df['return'].round(2),
            texttemplate='%{text}%',
            textposition='auto',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}年</b><br>' +
                         '收益率: %{y:.2f}%<extra></extra>',
            showlegend=False
        ), row=row, col=col)

        # 添加0%基准线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

        # 添加关键指标汇总注释
        stats_text = f"""
        <b>投资分析</b><br>
        总收益率: {self.results['total_return']:.2f}%<br>
        年化收益率: {self.results['annual_return']:.2f}%<br>
        最大回撤: {self.results['max_drawdown']:.2f}%<br>
        夏普比率: {self.results['sharpe_ratio']:.3f}<br>
        年化波动率: {self.results['volatility']:.2f}%
        """

        fig.add_annotation(
            text=stats_text,
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10),
            align="right",
            row=row,
            col=col
        )

        # 更新子图布局
        fig.update_xaxes(title_text="年份", row=row, col=col)
        fig.update_yaxes(title_text="收益率 (%)", tickformat='.1f', row=row, col=col)

    def _add_summary_stats_subplot(self, fig, row, col):
        """添加汇总统计信息子图"""
        # 创建统计信息文本
        stats_text = f"""
        <b>回测汇总统计</b><br><br>
        <b>投入情况:</b><br>
        初始资金: ¥{self.results['initial_capital']:,.0f}<br>
        总投入: ¥{self.results['total_investment']:,.0f}<br>
        最终价值: ¥{self.results['final_value']:,.0f}<br><br>

        <b>收益表现:</b><br>
        总收益率: {self.results['total_return']:.2f}%<br>
        年化收益率: {self.results['annual_return']:.2f}%<br>
        最大回撤: {self.results['max_drawdown']:.2f}%<br>
        年化波动率: {self.results['volatility']:.2f}%<br>
        夏普比率: {self.results['sharpe_ratio']:.3f}<br><br>

        <b>交易统计:</b><br>
        交易天数: {self.results['trading_days']}天<br>
        再平衡次数: {self.results['rebalance_count']}次<br>
        """

        # 添加文本作为注释
        fig.add_annotation(
            text=stats_text,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
            align="center",
            row=row,
            col=col
        )

        # 隐藏坐标轴
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def _plot_portfolio_value(self):
        """绘制账户资金变化折线图"""
        # 准备数据
        df = pd.DataFrame({
            'date': self.daily_dates,
            'value': self.daily_values
        })
        df['date'] = pd.to_datetime(df['date'])

        # 创建图表
        fig = go.Figure()

        # 主线：总资产价值
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name='总资产价值',
            line=dict(color='#2E86AB', width=2),
            hovertemplate='<b>%{x}</b><br>' +
                         '总资产: ¥%{y:,.0f}<extra></extra>'
        ))

        # 准备悬停信息
        hover_texts = []
        for i, date in enumerate(df['date']):
            if date in self.daily_positions:
                pos_info = self.daily_positions[date]
                hover_text = f"<b>{date.strftime('%Y-%m-%d')}</b><br>"
                hover_text += f"总资产: ¥{df.loc[i, 'value']:,.0f}<br><br>"

                # ETF持仓信息
                for etf_code in self.etf_codes:
                    if etf_code in pos_info:
                        info = pos_info[etf_code]
                        hover_text += f"{etf_code}: ¥{info['value']:,.0f} ({info['weight']:.1%})<br>"

                # 现金信息
                if 'cash' in pos_info:
                    cash_info = pos_info['cash']
                    hover_text += f"现金: ¥{cash_info['value']:,.0f} ({cash_info['weight']:.1%})"

                hover_texts.append(hover_text)
            else:
                hover_texts.append(f"<b>{date.strftime('%Y-%m-%d')}</b><br>总资产: ¥{df.loc[i, 'value']:,.0f}")

        # 添加再平衡标记
        if self.rebalance_dates:
            rebalance_values = []
            for date in self.rebalance_dates:
                # 找到最接近的日期索引
                closest_idx = min(range(len(df)), key=lambda i: abs(df['date'][i] - date))
                rebalance_values.append(df.loc[closest_idx, 'value'])

            fig.add_trace(go.Scatter(
                x=self.rebalance_dates,
                y=rebalance_values,
                mode='markers',
                name='再平衡',
                marker=dict(color='red', size=8, symbol='triangle-down'),
                hovertemplate='<b>再平衡</b><br>日期: %{x}<br>资产: ¥%{y:,.0f}<extra></extra>'
            ))

        # 添加定投标记
        if self.dca_dates:
            dca_values = []
            for date in self.dca_dates:
                closest_idx = min(range(len(df)), key=lambda i: abs(df['date'][i] - date))
                dca_values.append(df.loc[closest_idx, 'value'])

            fig.add_trace(go.Scatter(
                x=self.dca_dates,
                y=dca_values,
                mode='markers',
                name='定投',
                marker=dict(color='green', size=8, symbol='circle'),
                hovertemplate='<b>定投</b><br>日期: %{x}<br>资产: ¥%{y:,.0f}<extra></extra>'
            ))

        # 更新布局
        fig.update_layout(
            title={
                'text': '账户资金变化',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='日期',
            yaxis_title='账户价值 (¥)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True
        )

        # 设置y轴格式
        fig.update_yaxes(tickformat=',.0f')

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"portfolio_value_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def _plot_returns(self):
        """绘制收益率折线图"""
        # 准备数据
        # 计算每日累计投入
        daily_investments = self._calculate_daily_cumulative_investment()

        # 使用每日累计投入计算收益率
        returns_data = []
        for value, daily_investment in zip(self.daily_values, daily_investments):
            if daily_investment > 0:
                return_pct = (value / daily_investment - 1) * 100
            else:
                return_pct = 0
            returns_data.append(return_pct)

        df = pd.DataFrame({
            'date': self.daily_dates,
            'return_pct': returns_data
        })
        df['date'] = pd.to_datetime(df['date'])

        # 创建图表
        fig = go.Figure()

        # 主线：累计收益率
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['return_pct'],
            mode='lines',
            name='累计收益率',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{x}</b><br>' +
                         '收益率: %{y:.2f}%<extra></extra>'
        ))

        # 添加0%基准线
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="盈亏平衡线", annotation_position="bottom right")

        # 准备悬停信息
        custom_data = []
        for i, date in enumerate(df['date']):
            if date in self.daily_positions:
                pos_info = self.daily_positions[date]
                custom_info = {
                    'date': date.strftime('%Y-%m-%d'),
                    'return_pct': df.loc[i, 'return_pct'],
                    'positions': {}
                }

                # ETF持仓信息
                for etf_code in self.etf_codes:
                    if etf_code in pos_info:
                        info = pos_info[etf_code]
                        custom_info['positions'][etf_code] = {
                            'value': info['value'],
                            'weight': info['weight']
                        }

                # 现金信息
                if 'cash' in pos_info:
                    custom_info['positions']['cash'] = {
                        'value': pos_info['cash']['value'],
                        'weight': pos_info['cash']['weight']
                    }

                custom_data.append(custom_info)

        # 更新悬停模板
        fig.update_traces(
            customdata=custom_data,
            hovertemplate='<b>%{customdata.date}</b><br>' +
                         '收益率: %{y:.2f}%<br><br>' +
                         '<b>持仓分布:</b><br>' +
                         '%{customdata.positions[511010].value:,.0f} (511010)<br>' +
                         '%{customdata.positions[510880].value:,.0f} (510880)<br>' +
                         '%{customdata.positions[518880].value:,.0f} (518880)<br>' +
                         '%{customdata.positions[513100].value:,.0f} (513100)<br>' +
                         '现金: %{customdata.positions.cash.value:,.0f}<extra></extra>'
        )

        # 更新布局
        fig.update_layout(
            title={
                'text': '累计收益率变化',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='日期',
            yaxis_title='收益率 (%)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=False
        )

        # 设置y轴格式
        fig.update_yaxes(tickformat='.1f')

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"returns_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def _plot_drawdown(self):
        """绘制回撤图"""
        # 准备数据
        value_series = pd.Series(self.daily_values, index=self.daily_dates)
        rolling_max = value_series.expanding().max()
        drawdown = (value_series - rolling_max) / rolling_max * 100

        df = pd.DataFrame({
            'date': drawdown.index,
            'drawdown': drawdown.values
        })
        df['date'] = pd.to_datetime(df['date'])

        # 创建图表
        fig = go.Figure()

        # 回撤线
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['drawdown'],
            mode='lines',
            name='回撤',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            hovertemplate='<b>%{x}</b><br>' +
                         '回撤: %{y:.2f}%<extra></extra>'
        ))

        # 标记最大回撤点
        max_dd_idx = df['drawdown'].idxmin()
        max_dd_value = df['drawdown'].min()

        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers',
            name='最大回撤',
            marker=dict(color='darkred', size=12, symbol='x'),
            hovertemplate='<b>最大回撤</b><br>' +
                         '日期: %{x}<br>' +
                         '回撤: %{y:.2f}%<extra></extra>'
        ))

        # 如果有回补完成，标记回补点
        if self.results['recovery_date'] is not None:
            recovery_date = self.results['recovery_date']
            recovery_idx = df.index[df['date'] == recovery_date].tolist()
            if recovery_idx:
                recovery_idx = recovery_idx[0]
                recovery_value = df.loc[recovery_idx, 'drawdown']

                fig.add_trace(go.Scatter(
                    x=[recovery_date],
                    y=[recovery_value],
                    mode='markers',
                    name='回补完成',
                    marker=dict(color='green', size=12, symbol='circle'),
                    hovertemplate='<b>回补完成</b><br>' +
                                 '日期: %{x}<br>' +
                                 '回撤: %{y:.2f}%<extra></extra>'
                ))

        # 更新布局
        fig.update_layout(
            title={
                'text': '回撤分析',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='日期',
            yaxis_title='回撤 (%)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=False
        )

        # 设置y轴格式
        fig.update_yaxes(tickformat='.1f')

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"drawdown_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def _plot_monthly_heatmap(self):
        """绘制月度收益率heatmap"""
        # 准备月度数据
        df = pd.DataFrame({
            'date': self.daily_dates,
            'value': self.daily_values
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # 计算月度收益率
        monthly_returns = []
        monthly_data = []
        etf_monthly_data = {etf: [] for etf in self.etf_codes}

        for (year, month), group in df.groupby(['year', 'month']):
            if len(group) > 1:
                start_date = group.iloc[0]['date']
                end_date = group.iloc[-1]['date']
                start_value = group.iloc[0]['value']
                end_value = group.iloc[-1]['value']

                # 计算月度独立收益率
                month_return = self._calculate_period_return(start_date, end_date, start_value, end_value, use_period_investment=False)

                monthly_returns.append(month_return)
                monthly_data.append({
                    'year': year,
                    'month': month,
                    'return': month_return
                })

                # 计算各ETF在该月的贡献度
                for etf_code in self.etf_codes:
                    etf_contributions = []
                    for date in group['date']:
                        if date in self.daily_positions:
                            pos_info = self.daily_positions[date]
                            if etf_code in pos_info:
                                etf_contributions.append(pos_info[etf_code]['value'])

                    if len(etf_contributions) > 1:
                        etf_start = etf_contributions[0]
                        etf_end = etf_contributions[-1]
                        etf_month_return = (etf_end / etf_start - 1) * 100
                        etf_monthly_data[etf_code].append(etf_month_return)
                    else:
                        etf_monthly_data[etf_code].append(0)

        if not monthly_data:
            print("没有足够的月度数据生成热力图")
            return

        # 创建数据框
        heatmap_df = pd.DataFrame(monthly_data)
        heatmap_pivot = heatmap_df.pivot(index='year', columns='month', values='return')

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='RdYlGn',
            text=heatmap_pivot.round(2).values,
            texttemplate='%{text}%',
            textfont=dict(size=10),
            hoverongaps=False,
            colorbar=dict(
                title="收益率 (%)",
                title_side="right"
            ),
            hovertemplate='年份: %{y}<br>月份: %{x}<br>收益率: %{z:.2f}%<extra></extra>'
        ))

        # 更新布局
        fig.update_layout(
            title={
                'text': '月度收益率热力图',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='月份',
            yaxis_title='年份',
            template='plotly_white',
            height=600,
            width=800
        )

        # 设置月份标签
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']
        fig.update_xaxes(ticktext=month_names, tickvals=list(range(1, 13)))

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"monthly_heatmap_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def _plot_annual_returns(self):
        """绘制年度收益率柱状图"""
        # 准备年度数据
        df = pd.DataFrame({
            'date': self.daily_dates,
            'value': self.daily_values
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        # 计算年度收益率
        annual_returns = []
        annual_data = []
        etf_annual_data = {etf: [] for etf in self.etf_codes}

        for year, group in df.groupby('year'):
            if len(group) > 1:
                start_date = group.iloc[0]['date']
                end_date = group.iloc[-1]['date']
                start_value = group.iloc[0]['value']
                end_value = group.iloc[-1]['value']

                # 计算年度独立收益率
                year_return = self._calculate_period_return(start_date, end_date, start_value, end_value, use_period_investment=False)

                annual_returns.append(year_return)
                annual_data.append({
                    'year': year,
                    'return': year_return
                })

                # 计算各ETF年度贡献度
                for etf_code in self.etf_codes:
                    etf_contributions = []
                    for date in group['date']:
                        if date in self.daily_positions:
                            pos_info = self.daily_positions[date]
                            if etf_code in pos_info:
                                etf_contributions.append(pos_info[etf_code]['value'])

                    if len(etf_contributions) > 1:
                        etf_start = etf_contributions[0]
                        etf_end = etf_contributions[-1]
                        etf_annual_return = (etf_end / etf_start - 1) * 100
                        etf_annual_data[etf_code].append(etf_annual_return)
                    else:
                        etf_annual_data[etf_code].append(0)

        if not annual_data:
            print("没有足够的年度数据生成柱状图")
            return

        # 创建数据框
        annual_df = pd.DataFrame(annual_data)

        # 确定颜色（正收益绿色，负收益红色）
        colors = ['green' if ret >= 0 else 'red' for ret in annual_df['return']]

        # 创建柱状图
        fig = go.Figure(data=[go.Bar(
            x=annual_df['year'],
            y=annual_df['return'],
            name='年度收益率',
            marker_color=colors,
            text=annual_df['return'].round(2),
            texttemplate='%{text}%',
            textposition='auto',
            hovertemplate='<b>%{x}年</b><br>' +
                         '收益率: %{y:.2f}%<br>' +
                         '期初: ¥%{customdata.start_value:,.0f}<br>' +
                         '期末: ¥%{customdata.end_value:,.0f}<extra></extra>',
            customdata=annual_df
        )])

        # 添加0%基准线
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="盈亏平衡线", annotation_position="bottom right")

        # 更新布局
        fig.update_layout(
            title={
                'text': '年度收益率分析',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='年份',
            yaxis_title='收益率 (%)',
            template='plotly_white',
            height=600,
            showlegend=False
        )

        # 设置y轴格式
        fig.update_yaxes(tickformat='.1f')

        # 显示图表
        fig.show()

        # 保存为HTML文件（如果启用）
        if self.save_html:
            filename = f"annual_returns_{self.start_date.strftime('%Y%m%d')}_to_{self.end_date.strftime('%Y%m%d')}.html"
            fig.write_html(filename)
            print(f"图表已保存为HTML文件: {filename}")

    def get_results(self) -> Dict:
        """获取回测结果"""
        return self.results
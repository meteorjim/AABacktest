import pandas as pd
import numpy as np
import datetime
import akshare as ak
from typing import List, Dict
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
                # print(f"从缓存加载数据: {file_path}")
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
                    # print(f"从缓存加载 {stock} 数据成功 (匹配文件: {os.path.basename(matching_cache)})")
                    # 从缓存数据中筛选所需时间范围
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    filtered_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                    # print(f"从缓存数据中筛选 {start_date} 到 {end_date} 的数据，共 {len(filtered_data)} 条记录")
                    return filtered_data

            # 如果没有匹配的缓存文件，尝试精确匹配
            data = _load_from_cache(cache_file_path)
            if data is not None and not data.empty:
                # print(f"从缓存加载 {stock} 数据成功")
                return data

        # 从akshare获取历史数据

        # 方法1: 尝试获取股票数据
        try:
            stock_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            if stock_df is not None and not stock_df.empty:
                data = stock_df
                print(f"找到了{stock} 的A股数据")
        except:
            pass

        # 方法2: 尝试获取ETF数据
        if data is None:
            try:
                etf_df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
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
        # 保持datetime索引格式，不转换为字符串，以支持pandas时间序列功能

        # 计算涨跌幅，使用安全的方法处理缺失数据
        data = data.sort_index()
        data['cr'] = safe_pct_change(data['close'], fillna=True)

        # 计算移动平均线
        if need_ma:
            data['MA5'] = data['close'].rolling(5).mean()
            data['MA13'] = data['close'].rolling(13).mean()
            data['MA21'] = data['close'].rolling(21).mean()
            data['MA60'] = data['close'].rolling(60).mean()
            # 只删除cr列的NaN，保留移动平均线的NaN
            result = data[['close', 'open', 'high', 'low', 'volume', 'cr', 'MA5', 'MA13', 'MA21', 'MA60']].dropna(subset=['cr'])
        else:
            result = data[['close', 'open', 'high', 'low', 'volume', 'cr']].dropna(subset=['cr'])

        # 保存到缓存（如果启用缓存且数据不为空）
        if use_cache and result is not None and not result.empty:
            _save_to_cache(result, cache_file_path)

        print(f"成功获取 {stock} 数据，共 {len(result)} 条记录")
        return result

    except Exception as e:
        print(f"获取 {stock} 数据时出错: {e}")
        # 如果akshare失败，尝试从缓存加载（如果启用缓存）
        if use_cache and not force_refresh:
            print(f"尝试从缓存加载 {stock} 数据...")
            cached_data = _load_from_cache(cache_file_path)
            if cached_data is not None and not cached_data.empty:
                print(f"从缓存加载 {stock} 数据成功")
                return cached_data
        return pd.DataFrame()


class PortfolioBacktester:
    """
    资产配置策略回测系统

    功能：
    - 支持多ETF资产配置回测
    - 支持定期再平衡（年/半年/月）
    - 计算各项性能指标
    - 生成资产走势图
    """

    def __init__(self,
                 etf_codes: List[str],
                 weights: List[float],
                 rebalance_freq: str = 'yearly',
                 start_date: str = None,
                 end_date: str = None,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.0005,
                 risk_free_rate: float = 0.02,
                 use_cache: bool = True,
                 cache_dir: str = 'data',
                 force_refresh: bool = False,
                 enable_dca: bool = False,
                 dca_amount: float = 10000,
                 dca_freq: str = 'yearly'):
        """
        初始化回测器

        Args:
            etf_codes: ETF代码列表
            weights: 对应权重列表（总和应为1）
            rebalance_freq: 再平衡频率 ('yearly', 'semi_annual', 'quarterly', 'monthly')
            start_date: 回测开始日期 (格式: 'YYYY-MM-DD')
            end_date: 回测结束日期 (格式: 'YYYY-MM-DD')
            initial_capital: 初始资金
            transaction_cost: 交易成本比例
            risk_free_rate: 无风险利率（用于计算夏普率）
            use_cache: 是否使用本地缓存，默认True
            cache_dir: 缓存目录，默认'data'
            force_refresh: 是否强制刷新缓存，默认False
            enable_dca: 是否启用定投，默认False
            dca_amount: 定投金额，默认10000
            dca_freq: 定投频率 ('yearly', 'monthly')，默认'yearly'
        """
        self.etf_codes = etf_codes
        self.weights = np.array(weights)
        self.rebalance_freq = rebalance_freq

        # 缓存相关参数
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh

        # 定投相关参数
        self.enable_dca = enable_dca
        self.dca_amount = dca_amount
        self.dca_freq = dca_freq

        # 验证权重
        if abs(np.sum(weights) - 1.0) > 0.001:
            raise ValueError("权重总和必须等于1")

        if len(etf_codes) != len(weights):
            raise ValueError("ETF代码数量与权重数量不匹配")

        # 设置日期范围
        if end_date is None:
            self.end_date = datetime.datetime.now()
        else:
            self.end_date = pd.to_datetime(end_date)

        if start_date is None:
            self.start_date = self.end_date - datetime.timedelta(days=365*5)  # 默认5年
        else:
            self.start_date = pd.to_datetime(start_date)

        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        # 回测结果
        self.portfolio_value = None
        self.returns = None
        self.metrics = {}
        self.rebalance_dates = []
        self.etf_monthly_values = None  # 存储每个ETF的月度价值

        # 动态生成ETF信息
        self.etf_info = self._generate_etf_info()

    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        根据再平衡频率获取再平衡日期列表
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        if self.rebalance_freq == 'yearly':
            # 每年最后一个交易日
            rebalance_dates = dates[dates.is_year_end]
        elif self.rebalance_freq == 'semi_annual':
            # 每半年最后一个交易日（6月和12月）
            rebalance_dates = dates[(dates.month == 6) | (dates.month == 12)]
            rebalance_dates = rebalance_dates[rebalance_dates.is_month_end]
        elif self.rebalance_freq == 'monthly':
            # 每月最后一个交易日
            rebalance_dates = dates[dates.is_month_end]
        elif self.rebalance_freq == 'quarterly':
            # 每季度最后一个交易日（3月、6月、9月、12月）
            rebalance_dates = dates[(dates.month == 3) | (dates.month == 6) | (dates.month == 9) | (dates.month == 12)]
            rebalance_dates = rebalance_dates[rebalance_dates.is_month_end]
        else:
            raise ValueError("再平衡频率必须是 'yearly', 'semi_annual', 'quarterly', 或 'monthly'")

        return rebalance_dates

    def get_dca_dates(self, trading_dates: List[pd.Timestamp] = None) -> List[pd.Timestamp]:
        """
        根据定投频率获取定投日期列表

        Args:
            trading_dates: 实际交易日期列表，如果提供则选择最近的交易日

        Returns:
            List[pd.Timestamp]: 定投日期列表
        """
        if not self.enable_dca:
            return []

        if self.dca_freq == 'yearly':
            # 每年定投（从第二年开始，避免与初始建仓冲突）
            dca_dates = []
            start_year = self.start_date.year + 1  # 从第二年开始定投
            for year in range(start_year, self.end_date.year + 1):
                # 选择每年1月3日作为定投目标日（避开节假日）
                target_date = pd.Timestamp(f"{year}-01-03")

                if trading_dates:
                    # 如果提供了交易日列表，找到最接近目标日期的交易日
                    target_dca_date = None
                    min_diff = float('inf')
                    for trade_date in trading_dates:
                        if trade_date.year == year and trade_date.month == 1:
                            diff = abs((trade_date - target_date).days)
                            if diff < min_diff:
                                min_diff = diff
                                target_dca_date = trade_date
                    if target_dca_date and target_dca_date <= self.end_date:
                        dca_dates.append(target_dca_date)
                else:
                    # 如果没有交易日列表，使用目标日期
                    if target_date <= self.end_date:
                        dca_dates.append(target_date)
            return dca_dates

        elif self.dca_freq == 'monthly':
            # 每月定投（跳过开始当月）
            dca_dates = []
            start_month = self.start_date.replace(day=1) + pd.DateOffset(months=1)  # 从下个月开始
            current_month = start_month

            while current_month <= self.end_date:
                # 选择每月的1号作为定投目标日
                target_date = current_month.replace(day=1)

                if trading_dates:
                    # 如果提供了交易日列表，找到最接近目标日期的交易日
                    target_dca_date = None
                    min_diff = float('inf')
                    for trade_date in trading_dates:
                        if (trade_date.year == target_date.year and
                            trade_date.month == target_date.month):
                            diff = abs((trade_date - target_date).days)
                            if diff < min_diff:
                                min_diff = diff
                                target_dca_date = trade_date
                    if target_dca_date and target_dca_date <= self.end_date:
                        dca_dates.append(target_dca_date)
                else:
                    # 如果没有交易日列表，使用目标日期
                    if target_date <= self.end_date:
                        dca_dates.append(target_date)

                # 移动到下个月
                current_month = current_month + pd.DateOffset(months=1)
            return dca_dates
        else:
            raise ValueError("定投频率必须是 'yearly' 或 'monthly'")

    def _should_rebalance(self, current_date, trading_dates, current_index) -> bool:
        """
        智能判断是否应该在当前交易日进行再平衡
        避免因节假日跳过再平衡操作

        Args:
            current_date: 当前交易日
            trading_dates: 所有交易日列表
            current_index: 当前交易日在列表中的索引

        Returns:
            bool: 是否需要再平衡
        """
        current_dt = pd.to_datetime(current_date)

        # 获取下一个交易日（如果存在）
        next_trading_date = trading_dates[current_index + 1] if current_index + 1 < len(trading_dates) else None
        next_trading_dt = pd.to_datetime(next_trading_date) if next_trading_date is not None else None

        if self.rebalance_freq == 'monthly':
            # 如果下一个交易日的月份变了，说明今天是本月最后一个交易日
            if next_trading_dt is None or next_trading_dt.month != current_dt.month:
                return True

        elif self.rebalance_freq == 'quarterly':
            # 每季度最后一个交易日（3月、6月、9月、12月）
            if current_dt.month in [3, 6, 9, 12]:
                if next_trading_dt is None or next_trading_dt.month != current_dt.month:
                    return True

        elif self.rebalance_freq == 'yearly':
            # 如果下一个交易日的年份变了，说明今天是本年最后一个交易日
            if next_trading_dt is None or next_trading_dt.year != current_dt.year:
                return True

        elif self.rebalance_freq == 'semi_annual':
            # 每半年最后一个交易日（6月和12月）
            if current_dt.month in [6, 12]:
                if next_trading_dt is None or next_trading_dt.month != current_dt.month:
                    return True

        return False

    def fetch_etf_data(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有ETF的历史数据
        """
        etf_data = {}

        for etf_code in self.etf_codes:
            try:
                data = get_price_akshare(
                    stock=etf_code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    force_refresh=self.force_refresh
                )
                if data.empty:
                    print(f"警告: {etf_code} 没有获取到数据")
                    continue

                etf_data[etf_code] = data

            except Exception as e:
                print(f"获取 {etf_code} 数据时出错: {e}")
                continue

        if not etf_data:
            raise ValueError("未能获取任何ETF数据")

        return etf_data

    def _generate_etf_info(self) -> Dict[str, Dict]:
        """
        动态生成ETF信息映射，包括颜色和显示名称

        Returns:
            Dict: ETF信息字典，包含名称、颜色等
        """
        # 预定义的美观颜色池
        color_pool = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#393b79', '#637939', '#8c6d31', '#8c2d04', '#5254a3',
            '#843c39', '#e15759', '#7049aa', '#f28e2c', '#edc949',
            '#af7aa1', '#ff9d98', '#86bcb6', '#bab0ac', '#79706e'
        ]

        etf_info = {}
        for i, etf_code in enumerate(self.etf_codes):
            etf_info[etf_code] = {
                'name': f'ETF{etf_code}',  # 简洁的显示名称
                'color': color_pool[i % len(color_pool)],  # 循环使用颜色
                'index': i,  # 保持原始顺序
                'weight': self.weights[i]  # 对应权重
            }

        return etf_info

    def run_backtest(self) -> Dict:
        """
        运行回测 - 使用基于股数的正确算法
        """
        print("开始获取ETF数据...")

        try:
            etf_data = self.fetch_etf_data()

            # 验证ETF数据完整性
            if not etf_data:
                raise ValueError("未能获取任何ETF数据")

            print(f"成功获取 {len(etf_data)} 个ETF的数据")
            for etf_code, data in etf_data.items():
                print(f"  {etf_code}: {len(data)} 条记录")

        except Exception as e:
            print(f"获取ETF数据时出错: {e}")
            raise

        # 获取所有交易日
        all_dates = set()
        for data in etf_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(all_dates)

        # 初始化变量
        current_weights = self.weights.copy()
        rebalance_dates = self.get_rebalance_dates()
        dca_dates = self.get_dca_dates(trading_dates)  # 传入交易日列表获取定投日期
        shares = {}  # 持有的股数
        cash = self.initial_capital  # 现金

        # 用于跟踪定投记录
        self.dca_records = []

        # 记录每日组合价值和ETF价值
        daily_values = []
        daily_dates = []
        etf_daily_values = {etf_code: [] for etf_code in self.etf_codes}

        print("开始回测...")
        for i, date in enumerate(trading_dates):
            date_str = pd.to_datetime(date)

            # 第一天自动买入
            if i == 0:
                shares, cash = self.rebalance(shares, cash, etf_data, date, current_weights)
                print(f"在 {date_str} 初始建仓")

            # 检查是否需要定投
            elif self.enable_dca and date_str in dca_dates:
                # 验证定投日期是否为有效交易日（所有ETF都有数据）
                valid_dca_date = True
                for etf_code in self.etf_codes:
                    if date not in etf_data[etf_code].index:
                        valid_dca_date = False
                        break

                if valid_dca_date:
                    # 执行定投
                    cash += self.dca_amount  # 增加定投资金
                    shares, cash = self.rebalance(shares, cash, etf_data, date, current_weights)
                    self.dca_records.append({
                        'date': date_str,
                        'amount': self.dca_amount,
                        'total_portfolio_value': None  # 将在下面计算
                    })
                    print(f"在 {date_str} 执行定投 ¥{self.dca_amount:,.2f}")
                else:
                    print(f"定投日期 {date_str} 不是有效交易日，跳过定投")

            # 检查是否需要再平衡（使用智能判断，避免节假日跳过）
            elif self._should_rebalance(date, trading_dates, i):
                # 执行再平衡
                shares, cash = self.rebalance(shares, cash, etf_data, date, current_weights)
                self.rebalance_dates.append(date_str)
                # print(f"在 {date_str} 执行再平衡")

            # 计算当日组合价值
            portfolio_value = cash  # 初始为现金

            # 计算持有的ETF价值
            etf_values_today = {}
            for etf_code in self.etf_codes:
                etf_value = 0
                if etf_code in shares:
                    if date in etf_data[etf_code].index:
                        # 有当日价格数据，使用当日价格
                        price = etf_data[etf_code].loc[date, 'close']
                        etf_value = shares[etf_code] * price
                    elif i > 0 and len(etf_daily_values[etf_code]) > 0:
                        # 没有当日价格数据，使用前一天的价值（假设价格不变）
                        etf_value = etf_daily_values[etf_code][-1]
                        print(f"ETF {etf_code} 在 {date_str} 无数据，使用前一天价值 ¥{etf_value:,.0f}")
                    # 如果是第一天且没有数据，保持为0

                portfolio_value += etf_value
                etf_values_today[etf_code] = etf_value
                etf_daily_values[etf_code].append(etf_value)

            daily_values.append(portfolio_value)
            daily_dates.append(date_str)

            # 如果今天执行了定投，更新定投记录中的组合价值
            if self.enable_dca and self.dca_records and self.dca_records[-1]['date'] == date_str:
                self.dca_records[-1]['total_portfolio_value'] = portfolio_value

        # 创建结果DataFrame，确保日期索引一致性
        daily_dates = pd.to_datetime(daily_dates)
        self.portfolio_value = pd.DataFrame({
            'date': daily_dates,
            'portfolio_value': daily_values
        }).set_index('date')

        # 创建ETF价值DataFrame，确保索引一致性
        etf_value_df = pd.DataFrame(etf_daily_values, index=daily_dates)
        etf_value_df.index = pd.to_datetime(etf_value_df.index)

        # 验证数据完整性
        if len(self.portfolio_value) != len(etf_value_df):
            print("警告：投资组合价值和ETF价值数据长度不一致")

        # 保存ETF每日价值数据用于悬停信息
        self.etf_daily_values = etf_daily_values

        # 计算ETF月度价值
        self.etf_monthly_values = etf_value_df.resample('ME').last()

        # 计算收益率，使用安全的方法处理缺失数据
        if len(self.portfolio_value) > 1:
            self.portfolio_value['returns'] = safe_pct_change(self.portfolio_value['portfolio_value'], fillna=True)
        else:
            print("警告：数据不足，无法计算收益率")
            self.portfolio_value['returns'] = 0.0

        # 计算性能指标
        self.calculate_metrics()

        print("回测完成！")
        return self.get_results()

    def rebalance(self, shares, cash, etf_data, date, target_weights):
        """
        执行再平衡 - 卖出所有持仓，按目标权重重新买入，考虑交易成本
        """
        # 计算当前持仓的ETF价值
        etf_value = 0
        for etf_code in shares.keys():
            if date in etf_data[etf_code].index:
                price = etf_data[etf_code].loc[date, 'close']
                etf_value += shares[etf_code] * price

        # 计算当前总价值
        current_value = cash + etf_value

        # 计算交易成本（分别计算卖出和买入成本）
        # 卖出成本：基于ETF价值计算
        sell_cost = etf_value * self.transaction_cost if etf_value > 0 else 0

        # 卖出所有持仓后的现金
        cash_after_sell = cash + etf_value - sell_cost

        # 找出有数据的ETF及其原始权重
        available_etfs = []
        available_weights = []
        for i, etf_code in enumerate(etf_data.keys()):
            if date in etf_data[etf_code].index:
                available_etfs.append((etf_code, i))
                available_weights.append(target_weights[i])

        # 重新归一化权重，确保总和为100%
        total_available_weight = sum(available_weights)

        if total_available_weight > 0:
            normalized_weights = [w / total_available_weight for w in available_weights]

            # 计算买入成本：基于买入金额计算
            # 先计算总买入金额（可用现金）
            total_buy_amount = cash_after_sell
            buy_cost = total_buy_amount * self.transaction_cost if total_buy_amount > 0 else 0

            # 实际用于投资的资金
            reinvest_capital = cash_after_sell - buy_cost

            # 清空当前持仓
            new_shares = {}

            # 按重新归一化的权重分配资金
            for (etf_code, original_index), normalized_weight in zip(available_etfs, normalized_weights):
                price = etf_data[etf_code].loc[date, 'close']
                target_value = reinvest_capital * normalized_weight
                new_shares[etf_code] = target_value / price  # 买入股数

            # 再平衡后现金为0（全部资金用于买入有数据的ETF）
            cash = 0
        else:
            # 如果没有任何ETF有数据，保持全部现金
            new_shares = {}
            cash = cash_after_sell

        return new_shares, cash

    def calculate_metrics(self):
        """
        计算性能指标 - 支持定投功能
        """
        if self.portfolio_value is None:
            raise ValueError("请先运行回测")

        if self.portfolio_value.empty:
            raise ValueError("投资组合数据为空")

        returns = self.portfolio_value['returns'].dropna()
        values = self.portfolio_value['portfolio_value']

        if len(values) == 0:
            raise ValueError("投资组合价值数据为空")

        if len(returns) == 0:
            print("警告：没有有效的收益率数据，部分指标可能不准确")

        # 计算总投资金额（初始资金 + 定投金额）
        total_invested = self.initial_capital
        if self.enable_dca and hasattr(self, 'dca_records'):
            total_invested += sum(record['amount'] for record in self.dca_records)

        # 总收益率 - 根据是否启用定投使用不同的计算方法
        if self.enable_dca:
            # 定投模式：基于总投资金额计算收益率
            total_return = (values.iloc[-1] - total_invested) / total_invested if total_invested > 0 else 0
        else:
            # 普通模式：基于初始资金计算收益率
            total_return = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]

        # 年化收益率
        days_held = (values.index[-1] - values.index[0]).total_seconds() / (24 * 60 * 60)
        annual_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 else 0

        # 最大回撤
        cumulative_max = values.cummax()
        drawdown = (cumulative_max - values) / cumulative_max
        max_drawdown = drawdown.max()

        # 夏普率
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 0 else 0

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 计算定投相关指标
        dca_count = len(self.dca_records) if self.enable_dca and hasattr(self, 'dca_records') else 0
        total_dca_amount = sum(record['amount'] for record in self.dca_records) if self.enable_dca and hasattr(self, 'dca_records') else 0

        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'final_value': values.iloc[-1],
            'rebalance_count': len(self.rebalance_dates),
            'total_invested': total_invested,
            'dca_count': dca_count,
            'total_dca_amount': total_dca_amount
        }

    def get_results(self) -> Dict:
        """
        获取回测结果
        """
        return {
            'metrics': self.metrics,
            'portfolio_value': self.portfolio_value,
            'rebalance_dates': self.rebalance_dates
        }

    def print_results(self):
        """
        打印回测结果 - 支持定投信息显示
        """
        if not self.metrics:
            print("没有可显示的结果")
            return

        print("\n" + "="*50)
        print("资产配置回测结果")
        print("="*50)
        print(f"ETF组合: {', '.join(self.etf_codes)}")
        print(f"权重配置: {', '.join([f'{w:.1%}' for w in self.weights])}")
        print(f"再平衡频率: {self.rebalance_freq}")
        print(f"回测期间: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")

        # 显示定投配置信息
        if self.enable_dca:
            print(f"定投配置: {self.dca_freq} 定投 ¥{self.dca_amount:,.2f}")

        print("-"*50)
        print(f"初始资金: ¥{self.initial_capital:,.2f}")

        # 显示定投相关信息
        if self.enable_dca and self.metrics.get('total_dca_amount', 0) > 0:
            print(f"定投总额: ¥{self.metrics['total_dca_amount']:,.2f}")
            print(f"定投次数: {self.metrics['dca_count']} 次")
            print(f"总投资金额: ¥{self.metrics['total_invested']:,.2f}")

        print(f"最终价值: ¥{self.metrics['final_value']:,.2f}")
        print(f"总收益率: {self.metrics['total_return']:.2%}")
        print(f"年化收益率: {self.metrics['annual_return']:.2%}")
        print(f"最大回撤: {self.metrics['max_drawdown']:.2%}")
        print(f"夏普率: {self.metrics['sharpe_ratio']:.3f}")
        print(f"年化波动率: {self.metrics['volatility']:.2%}")
        print(f"再平衡次数: {self.metrics['rebalance_count']}")
        print("="*50)

    def _calculate_monthly_returns(self) -> pd.Series:
        """
        计算考虑定投因素的月度收益率

        Returns:
            pd.Series: 月度收益率序列，考虑了定投的额外资金投入
        """
        if self.portfolio_value is None or self.portfolio_value.empty:
            return pd.Series([], dtype=float)

        monthly_data = self.portfolio_value.resample('ME').last()
        if len(monthly_data) < 2:
            return pd.Series([], dtype=float)

        monthly_returns = []

        for i, (date, portfolio_value) in enumerate(monthly_data['portfolio_value'].items()):
            if i == 0:
                # 第一个月的收益率为0
                monthly_returns.append(0.0)
            else:
                prev_date = monthly_data.index[i-1]
                prev_portfolio_value = monthly_data['portfolio_value'].iloc[i-1]

                # 检查该月是否有定投
                monthly_dca = 0
                dca_in_month = False
                if self.enable_dca and hasattr(self, 'dca_records'):
                    for record in self.dca_records:
                        # 如果定投发生在当前月份
                        if prev_date < record['date'] <= date:
                            monthly_dca += record['amount']
                            dca_in_month = True

                if dca_in_month:
                    # 如果有定投，使用基于ETF加权收益的计算方法
                    monthly_return = self._calculate_monthly_return_with_dca(date, prev_date, monthly_dca)
                else:
                    # 如果没有定投，使用标准计算
                    if prev_portfolio_value > 0:
                        net_change = portfolio_value - prev_portfolio_value
                        monthly_return = (net_change / prev_portfolio_value) * 100
                    else:
                        monthly_return = 0.0

                monthly_returns.append(monthly_return)

        return pd.Series(monthly_returns, index=monthly_data.index)

    def _calculate_monthly_return_with_dca(self, current_date: pd.Timestamp, prev_date: pd.Timestamp, dca_amount: float) -> float:
        """
        计算有定投月份的收益率，基于ETF的加权收益

        Args:
            current_date: 当前月末日期
            prev_date: 上月末日期
            dca_amount: 定投金额

        Returns:
            float: 月度收益率（百分比）
        """
        try:
            # 获取各个ETF在该月的收益率
            etf_monthly_returns = {}
            for etf_code in self.etf_codes:
                try:
                    # 获取ETF价格数据
                    price_data = get_price_akshare(
                        stock=etf_code,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        need_ma=False,
                        use_cache=self.use_cache,
                        cache_dir=self.cache_dir,
                        force_refresh=self.force_refresh
                    )

                    if not price_data.empty:
                        # 计算月末价格和收益率
                        monthly_prices = price_data.resample('ME').last()

                        # 查找该月的收益率
                        if current_date in monthly_prices.index and prev_date in monthly_prices.index:
                            prev_price = monthly_prices.loc[prev_date, 'close']
                            curr_price = monthly_prices.loc[current_date, 'close']
                            if prev_price > 0:
                                etf_return = ((curr_price - prev_price) / prev_price) * 100
                                etf_monthly_returns[etf_code] = etf_return
                            else:
                                etf_monthly_returns[etf_code] = 0.0
                        else:
                            etf_monthly_returns[etf_code] = 0.0
                except Exception as e:
                    print(f"计算{etf_code}月度收益率时出错: {e}")
                    etf_monthly_returns[etf_code] = 0.0

            # 计算投资组合的加权收益率
            weighted_return = 0.0
            for etf_code in self.etf_codes:
                if etf_code in etf_monthly_returns and etf_code in self.etf_info:
                    weight = self.etf_info[etf_code]['weight']
                    weighted_return += etf_monthly_returns[etf_code] * weight

            return weighted_return

        except Exception as e:
            print(f"计算定投月份收益率时出错: {e}")
            # 回退到简化计算
            return 0.0

    def _calculate_cumulative_returns(self) -> pd.Series:
        """
        计算考虑定投因素的累计收益率

        Returns:
            pd.Series: 累计收益率序列，考虑了定投的额外资金投入
        """
        if self.portfolio_value is None or self.portfolio_value.empty:
            return pd.Series([], dtype=float)

        returns_series = []

        for i, (date, portfolio_value) in enumerate(self.portfolio_value['portfolio_value'].items()):
            if i == 0:
                # 第一天的收益率为0
                returns_series.append(0.0)
            else:
                # 计算到该日期为止的总投入资金
                total_invested = self.initial_capital

                if self.enable_dca and hasattr(self, 'dca_records'):
                    for record in self.dca_records:
                        if record['date'] <= date:
                            total_invested += record['amount']

                # 计算真实的累计收益率：（当前价值 - 总投入） / 总投入
                if total_invested > 0:
                    cumulative_return = ((portfolio_value - total_invested) / total_invested) * 100
                else:
                    cumulative_return = 0.0

                returns_series.append(cumulative_return)

        return pd.Series(returns_series, index=self.portfolio_value.index)

    def plot_results(self, show_benchmark: bool = False, benchmark_code: str = 'SPY'):
        """
        绘制交互式资产走势图

        Args:
            show_benchmark: 是否显示基准对比
            benchmark_code: 基准ETF代码
        """
        if self.portfolio_value is None:
            print("没有可绘制的数据")
            return

        # 准备ETF数据用于增强的悬停信息
        etf_daily_hover_data = self._prepare_etf_hover_data()

        # 创建子图 - 增加一行用于收益率图
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('投资组合价值走势', '资产收益率', '月度收益率', '回撤分析'),
            vertical_spacing=0.05,
            horizontal_spacing=0.02
        )

        # 1. 投资组合价值曲线 - 增强悬停信息
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_value.index,
                y=self.portfolio_value['portfolio_value'],
                mode='lines',
                name='投资组合价值',
                line=dict(color='blue', width=2),
                customdata=etf_daily_hover_data,
                hovertemplate=self._generate_portfolio_hover_template(),
                hoverlabel=dict(bgcolor="white", font_size=12)
            ),
            row=1, col=1
        )

        # 添加再平衡标记线
        for i, rebalance_date in enumerate(self.rebalance_dates):
            fig.add_shape(
                type="line",
                x0=rebalance_date, x1=rebalance_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.6,
                row=1, col=1
            )

        # 添加基准对比
        if show_benchmark:
            try:
                benchmark_data = get_price_akshare(
                    stock=benchmark_code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    force_refresh=self.force_refresh
                )
                if not benchmark_data.empty:
                    # 标准化基准数据到相同的初始值
                    benchmark_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0] * self.initial_capital

                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_normalized.index,
                            y=benchmark_normalized,
                            mode='lines',
                            name=f'{benchmark_code} 基准',
                            line=dict(color='orange', width=1.5),
                            opacity=0.8,
                            hovertemplate='日期: %{x}<br>基准价值: ¥%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            except Exception as e:
                print(f"无法获取基准数据 {benchmark_code}: {e}")

        # 2. 资产收益率折线图 - 显示累计收益率（修正：考虑定投因素）
        cumulative_returns = self._calculate_cumulative_returns()

        # 准备ETF收益率数据用于悬停
        etf_returns_data = self._prepare_etf_returns_data()

        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='累计收益率',
                line=dict(color='green', width=2.5),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                customdata=etf_returns_data,
                hovertemplate=self._generate_returns_hover_template(),
                hoverlabel=dict(bgcolor="white", font_size=12)
            ),
            row=2, col=1
        )

        # 添加零线
        fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)

        # 3. 月度收益率柱状图（修正：使用考虑定投的月度收益率）
        monthly_returns = self._calculate_monthly_returns()

        colors = ['green' if r > 0 else 'red' for r in monthly_returns]

        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns * 100,
                name='月度收益率',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='日期: %{x}<br>收益率: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )

        # 添加零线
        fig.add_hline(y=0, line=dict(color='black', width=1), row=3, col=1)

        # 4. 回撤图
        portfolio_values = self.portfolio_value['portfolio_value']
        cumulative_max = portfolio_values.cummax()
        drawdown = (cumulative_max - portfolio_values) / cumulative_max

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                name='回撤',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)',
                hovertemplate='日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>'
            ),
            row=4, col=1
        )

        # 更新布局
        fig.update_layout(
            title=dict(
                text="交互式投资组合分析报告",
                x=0.5,
                font=dict(size=18, family="Arial, sans-serif")
            ),
            height=1200,  # 增加高度以容纳4个图
            width=1000,
            showlegend=True,
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 更新x轴和y轴标题
        fig.update_xaxes(title_text="日期", row=4, col=1)
        fig.update_yaxes(title_text="投资组合价值 (¥)", row=1, col=1)
        fig.update_yaxes(title_text="累计收益率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="月度收益率 (%)", row=3, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=4, col=1)

        # 显示图表
        fig.show()

    def plot_monthly_heatmap(self):
        """
        绘制交互式月度收益率热力图和年度收益图表，显示ETF详细信息
        """
        if self.portfolio_value is None:
            print("没有可绘制的数据")
            return

        # 准备数据
        heatmap_data = self._prepare_interactive_heatmap_data()
        if heatmap_data is None:
            return

        # 准备年度收益数据
        yearly_data = self._prepare_yearly_returns_data()
        if yearly_data is None:
            print("无法准备年度收益数据，只显示热力图")
            # 如果年度数据失败，只显示热力图
            return self._plot_heatmap_only(heatmap_data)

        # 创建子图布局
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "heatmap", "colspan": 2}, None],
                   [{"type": "bar"}, {"type": "table"}]],
            subplot_titles=('月度收益率热力图', '年度收益率分析', '年度收益统计表'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            row_heights=[0.7, 0.3]
        )

        # 1. 月度收益率热力图 (占据第一行两列)
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data['values'],
                x=heatmap_data['months'],
                y=heatmap_data['years'],
                colorscale='RdYlGn',
                zmid=0,
                zmin=heatmap_data['vmin'],
                zmax=heatmap_data['vmax'],
                text=heatmap_data['values'],  # 使用数值数据用于显示
                texttemplate="%{text:.1f}%",  # 在热力图上显示百分比数字
                textfont=dict(size=10, color="black"),  # 设置字体大小和颜色
                hovertemplate='%{hovertext}<extra></extra>',  # 使用详细悬停文本
                hovertext=heatmap_data['text'],  # 设置详细的悬停文本
                colorbar=dict(
                    title="月度收益率 (%)",
                    tickmode="array",
                    tickvals=[heatmap_data['vmin'], 0, heatmap_data['vmax']],
                    ticktext=[f"{heatmap_data['vmin']:.1f}%", "0%", f"{heatmap_data['vmax']:.1f}%"],
                    x=1.02,  # 调整颜色条位置
                    len=0.7   # 调整颜色条长度
                ),
                showscale=True
            ),
            row=1, col=1
        )

        # 2. 年度收益率柱状图
        colors = ['green' if val > 0 else 'red' for val in yearly_data['returns']]
        fig.add_trace(
            go.Bar(
                x=yearly_data['years'],
                y=yearly_data['returns'],
                name='年度收益率',
                marker_color=colors,
                opacity=0.8,
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=yearly_data['hover_texts'],  # 使用详细的悬停文本
                text=yearly_data['returns_text'],
                textposition='outside',
                textfont=dict(size=10)
            ),
            row=2, col=1
        )

        # 添加零线
        fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)

        # 3. 年度收益统计表
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['年份', '收益率', '最终价值', '定投金额'],
                    fill_color='lightblue',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[
                        yearly_data['years'],
                        yearly_data['returns_text'],
                        yearly_data['final_values_text'],
                        yearly_data['dca_amounts_text']
                    ],
                    fill_color='white',
                    align='center',
                    font=dict(size=11),
                    height=25
                )
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            title=dict(
                text="交互式投资组合收益分析报告",
                x=0.5,
                font=dict(size=18, family="Arial, sans-serif")
            ),
            width=1200,
            height=900,
            showlegend=False,
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 更新子图坐标轴
        # 热力图坐标轴
        fig.update_xaxes(
            title_text="月份",
            tickmode='array',
            tickvals=list(range(12)),
            ticktext=['1月', '2月', '3月', '4月', '5月', '6月',
                      '7月', '8月', '9月', '10月', '11月', '12月'],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="年份",
            autorange="reversed",
            row=1, col=1
        )

        # 年度收益柱状图坐标轴
        fig.update_xaxes(title_text="年份", row=2, col=1)
        fig.update_yaxes(title_text="年度收益率 (%)", row=2, col=1)

        # 隐藏表格的坐标轴
        fig.update_xaxes(showgrid=False, showticklabels=False, row=2, col=2)
        fig.update_yaxes(showgrid=False, showticklabels=False, row=2, col=2)

        # 显示图表
        fig.show()

        return fig

    def _plot_heatmap_only(self, heatmap_data):
        """
        只显示热力图的备用函数

        Args:
            heatmap_data: 热力图数据
        """
        # 创建只包含热力图的布局
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data['values'],
            x=heatmap_data['months'],
            y=heatmap_data['years'],
            colorscale='RdYlGn',
            zmid=0,
            zmin=heatmap_data['vmin'],
            zmax=heatmap_data['vmax'],
            text=heatmap_data['values'],  # 使用数值数据用于显示
            texttemplate="%{text:.1f}%",  # 在热力图上显示百分比数字
            textfont=dict(size=10, color="black"),  # 设置字体大小和颜色
            hovertemplate='%{hovertext}<extra></extra>',  # 使用详细悬停文本
            hovertext=heatmap_data['text'],  # 设置详细的悬停文本
            colorbar=dict(
                title="月度收益率 (%)",
                tickmode="array",
                tickvals=[heatmap_data['vmin'], 0, heatmap_data['vmax']],
                ticktext=[f"{heatmap_data['vmin']:.1f}%", "0%", f"{heatmap_data['vmax']:.1f}%"]
            )
        ))

        # 设置布局
        fig.update_layout(
            title=dict(
                text="交互式月度收益率热力图",
                x=0.5,
                font=dict(size=16, family="Arial, sans-serif")
            ),
            xaxis=dict(
                title="月份",
                tickmode='array',
                tickvals=list(range(12)),
                ticktext=['1月', '2月', '3月', '4月', '5月', '6月',
                          '7月', '8月', '9月', '10月', '11月', '12月']
            ),
            yaxis=dict(
                title="年份",
                autorange="reversed"
            ),
            width=900,
            height=600,
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 显示图表
        fig.show()
        return fig

    def _prepare_yearly_returns_data(self) -> Dict:
        """
        准备年度收益数据，包含ETF明细悬停信息

        Returns:
            Dict: 包含年度收益数据的字典
        """
        try:
            if self.portfolio_value is None or self.portfolio_value.empty:
                print("投资组合数据为空，无法生成年度收益数据")
                return None

            # 按年度分组计算收益
            yearly_data = self.portfolio_value.resample('YS').last()

            if len(yearly_data) < 1:
                print("投资组合数据不足，无法生成年度收益数据")
                return None

            # 获取ETF价格数据用于计算年度收益
            etf_yearly_returns = {}
            for etf_code in self.etf_codes:
                try:
                    price_data = get_price_akshare(
                        stock=etf_code,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        need_ma=False,
                        use_cache=self.use_cache,
                        cache_dir=self.cache_dir,
                        force_refresh=self.force_refresh
                    )

                    if not price_data.empty:
                        # 计算年末价格和收益率，使用安全的方法
                        yearly_prices = price_data.resample('YS').last()
                        returns = safe_pct_change(yearly_prices['close'], fillna=False)  # 不填充NaN，后面会处理

                        # 为第一年计算基于初始价格的收益率（只计算价格增长，不包括投资本金）
                        if len(yearly_prices) >= 1:
                            # 计算第一年的收益率（纯粹的价格增长）
                            first_year_return = (yearly_prices.iloc[0] / price_data['close'].iloc[0] - 1) * 100
                            # 确保赋值为单个数值，避免"setting an array element with a sequence"错误
                            if isinstance(first_year_return, (pd.Series, np.ndarray)):
                                if len(first_year_return) > 0:
                                    returns.iloc[0] = float(first_year_return.iloc[0] if hasattr(first_year_return, 'iloc') else first_year_return[0]) / 100
                                else:
                                    returns.iloc[0] = 0.0
                            else:
                                returns.iloc[0] = float(first_year_return) / 100  # 转换为小数形式

                        etf_yearly_returns[etf_code] = returns
                except Exception as e:
                    print(f"获取{etf_code}年度价格数据时出错: {e}")
                    continue

            years = []
            returns = []
            returns_text = []
            final_values = []
            final_values_text = []
            dca_amounts = []
            dca_amounts_text = []
            hover_texts = []  # 新增：悬停文本列表

            # 计算每年的收益率和相关数据
            for i, (year_end, row) in enumerate(yearly_data.iterrows()):
                year = year_end.year
                years.append(str(year))

                # 计算该年收益率
                if i == 0:
                    # 第一年，基于初始资金计算
                    initial_value = self.initial_capital
                    # 检查是否有定投
                    yearly_dca = 0
                    if self.enable_dca and hasattr(self, 'dca_records'):
                        for record in self.dca_records:
                            if record['date'].year == year:
                                yearly_dca += record['amount']
                    total_invested = initial_value + yearly_dca
                else:
                    # 后续年份，基于年初价值计算
                    prev_year_end = yearly_data.index[i-1]
                    if prev_year_end in self.portfolio_value.index:
                        initial_value = self.portfolio_value.loc[prev_year_end, 'portfolio_value']
                    else:
                        # 如果找不到年初数据，使用上一年的年末数据
                        initial_value = yearly_data.iloc[i-1]['portfolio_value']

                    # 计算该年定投
                    yearly_dca = 0
                    if self.enable_dca and hasattr(self, 'dca_records'):
                        for record in self.dca_records:
                            if record['date'].year == year:
                                yearly_dca += record['amount']
                    total_invested = initial_value + yearly_dca

                final_value = row['portfolio_value']
                # 修正年度收益率计算：只计算投资增长的收益，不包括定投资金
                # 年度收益率应该基于年初价值计算，而不是总投资金额
                yearly_return = ((final_value - initial_value - yearly_dca) / initial_value) * 100 if initial_value > 0 else 0

                returns.append(yearly_return)
                returns_text.append(f"{yearly_return:+.2f}%")
                final_values.append(final_value)
                final_values_text.append(f"¥{final_value:,.0f}")
                dca_amounts.append(yearly_dca)
                dca_amounts_text.append(f"¥{yearly_dca:,.0f}" if yearly_dca > 0 else "¥0")

                # 生成详细的悬停文本
                hover_text = self._generate_yearly_hover_text(
                    year, yearly_return, etf_yearly_returns,
                    initial_value, final_value, yearly_dca
                )
                hover_texts.append(hover_text)

            return {
                'years': years,
                'returns': returns,
                'returns_text': returns_text,
                'final_values': final_values,
                'final_values_text': final_values_text,
                'dca_amounts': dca_amounts,
                'dca_amounts_text': dca_amounts_text,
                'hover_texts': hover_texts  # 新增：悬停文本
            }

        except Exception as e:
            print(f"准备年度收益数据时出错: {e}")
            return None

    def _generate_yearly_hover_text(self, year: int, portfolio_return: float,
                                etf_returns: Dict[str, pd.Series],
                                initial_value: float, final_value: float,
                                yearly_dca: float) -> str:
        """
        生成年度收益柱状图的悬停文本，包含ETF明细

        Args:
            year: 年份
            portfolio_return: 投资组合年度收益率
            etf_returns: ETF年度收益率数据
            initial_value: 年初价值
            final_value: 年末价值
            yearly_dca: 年度定投金额

        Returns:
            str: 格式化的悬停文本
        """
        lines = [f"<b>{year}年投资组合收益报告</b>"]
        lines.append("")
        lines.append(f"💰 年初价值: ¥{initial_value:,.0f}")
        lines.append(f"💰 年末价值: ¥{final_value:,.0f}")
        if yearly_dca > 0:
            lines.append(f"💰 定投金额: ¥{yearly_dca:,.0f}")
        lines.append(f"📈 年度收益率: <b>{portfolio_return:+.2f}%</b>")
        lines.append("")
        lines.append("📊 <b>ETF年度明细:</b>")

        # 按原始顺序显示ETF信息
        for etf_code in self.etf_codes:
            if etf_code in etf_returns and etf_code in self.etf_info:
                etf_info = self.etf_info[etf_code]
                etf_return_series = etf_returns[etf_code]

                # 查找该年度的收益率
                yearly_etf_return = 0.0  # 默认为0%，表示无变化
                weight = etf_info['weight'] * 100  # 确保weight始终有值
                contribution = 0.0

                # 尝试多种方式查找年度收益率
                if not etf_return_series.empty:
                    # 方法1: 直接按年份查找
                    for idx, ret_val in etf_return_series.items():
                        if hasattr(idx, 'year') and idx.year == year:
                            if not pd.isna(ret_val):
                                yearly_etf_return = ret_val * 100
                                contribution = yearly_etf_return * etf_info['weight']
                                break

                    # 方法2: 如果方法1失败，尝试按字符串年份查找
                    if yearly_etf_return == 0.0:
                        for idx, ret_val in etf_return_series.items():
                            if str(idx) == str(year):
                                if not pd.isna(ret_val):
                                    yearly_etf_return = ret_val * 100
                                    contribution = yearly_etf_return * etf_info['weight']
                                    break

                lines.append(f"├─ {etf_info['name']}: <b>{yearly_etf_return:+.2f}%</b> | 占比{weight:.0f}% | 贡献{contribution:+.2f}%")

        # 添加统计信息
        valid_returns = []
        valid_etf_indices = []
        for etf_code in self.etf_codes:
            if etf_code in etf_returns:
                etf_return_series = etf_returns[etf_code]
                etf_return = 0

                # 尝试多种方式查找年度收益率
                if not etf_return_series.empty:
                    # 方法1: 直接按年份查找
                    for idx, ret_val in etf_return_series.items():
                        if hasattr(idx, 'year') and idx.year == year:
                            if not pd.isna(ret_val):
                                etf_return = ret_val * 100
                                break

                    # 方法2: 如果方法1失败，尝试按字符串年份查找
                    if etf_return == 0.0:
                        for idx, ret_val in etf_return_series.items():
                            if str(idx) == str(year):
                                if not pd.isna(ret_val):
                                    etf_return = ret_val * 100
                                    break

                # 总是添加到有效收益列表中，即使为0%（表示无变化）
                valid_returns.append(etf_return)
                valid_etf_indices.append(self.etf_codes.index(etf_code))

        if valid_returns:
            best_return = max(valid_returns)
            worst_return = min(valid_returns)
            best_etf_idx = valid_returns.index(best_return)
            worst_etf_idx = valid_returns.index(worst_return)

            lines.append("")
            lines.append(f"📈 当年最佳: <b>{self.etf_info[self.etf_codes[valid_etf_indices[best_etf_idx]]]['name']} ({best_return:+.2f}%)</b>")
            lines.append(f"📉 当年最差: <b>{self.etf_info[self.etf_codes[valid_etf_indices[worst_etf_idx]]]['name']} ({worst_return:+.2f}%)</b>")

        return "<br>".join(lines)

    def _prepare_interactive_heatmap_data(self) -> Dict:
        """
        准备交互式热力图数据

        Returns:
            Dict: 包含热力图所需所有数据的字典
        """
        try:
            # 检查portfolio_value是否存在
            if self.portfolio_value is None or self.portfolio_value.empty:
                print("投资组合数据为空，无法生成热力图")
                return None

            # 计算投资组合月度收益率（修正：考虑定投因素）
            monthly_data = self.portfolio_value.resample('ME').last()
            if len(monthly_data) < 2:
                print("投资组合数据不足，无法生成热力图")
                return None

            # 修正月度收益率计算，考虑定投的额外资金投入
            portfolio_returns = self._calculate_monthly_returns()
            if portfolio_returns.empty:
                print("无法计算投资组合收益率，无法生成热力图")
                return None

            # 创建年-月矩阵
            years = portfolio_returns.index.year.unique()
            months = ['1月', '2月', '3月', '4月', '5月', '6月',
                     '7月', '8月', '9月', '10月', '11月', '12月']

            # 初始化数据矩阵
            heatmap_values = pd.DataFrame(index=years, columns=range(12))
            heatmap_text = pd.DataFrame(index=years, columns=range(12))

            # 计算ETF月度收益率用于悬停信息（基于价格数据）
            etf_monthly_returns = {}
            for etf_code in self.etf_codes:
                try:
                    # 获取ETF价格数据，与文字报告保持一致
                    price_data = get_price_akshare(
                        stock=etf_code,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        need_ma=False,
                        use_cache=self.use_cache,
                        cache_dir=self.cache_dir,
                        force_refresh=self.force_refresh
                    )

                    if not price_data.empty:
                        # 计算月末价格和收益率，使用安全的方法
                        monthly_prices = price_data.resample('ME').last()
                        returns = safe_pct_change(monthly_prices['close'], fillna=False)  # 不填充NaN，后面会处理
                        etf_monthly_returns[etf_code] = returns
                except Exception as e:
                    print(f"获取{etf_code}价格数据时出错: {e}")
                    continue

            # 填充热力图数据
            for date, return_val in portfolio_returns.items():
                year = date.year
                month = date.month - 1  # 转换为0-11索引

                # 设置热力图值（return_val已经是百分比形式）
                heatmap_values.loc[year, month] = return_val

                # 生成悬停文本（return_val已经是百分比形式）
                hover_text = self._generate_hover_text(
                    date, return_val, etf_monthly_returns
                )
                heatmap_text.loc[year, month] = hover_text

            # 填充缺失值
            heatmap_values = heatmap_values.fillna(0)
            heatmap_text = heatmap_text.fillna('无数据')

            # 计算颜色范围，确保关于0对称
            abs_max = max(abs(heatmap_values.min().min()), abs(heatmap_values.max().max()))
            vmin = -abs_max
            vmax = abs_max

            return {
                'values': heatmap_values.values,
                'months': months,
                'years': years,
                'text': heatmap_text.values,
                'vmin': vmin,
                'vmax': vmax
            }

        except Exception as e:
            print(f"准备热力图数据时出错: {e}")
            return None

    def _generate_hover_text(self, date: pd.Timestamp, portfolio_return: float,
                           etf_returns: Dict[str, pd.Series]) -> str:
        """
        生成热力图悬停文本

        Args:
            date: 实际的月末日期
            portfolio_return: 投资组合收益率
            etf_returns: ETF收益率数据

        Returns:
            str: 格式化的悬停文本
        """
        year = date.year
        month = date.month
        lines = [f"{year}年{month}月 总收益: <b>{portfolio_return:+.2f}%</b>"]
        lines.append("")  # 空行
        lines.append("📊 <b>ETF明细:</b>")

        # 按原始顺序显示ETF信息
        for etf_code in self.etf_codes:
            if etf_code in etf_returns and etf_code in self.etf_info:
                etf_info = self.etf_info[etf_code]
                etf_return_series = etf_returns[etf_code]

                # 直接使用传入的date查询对应月份的收益率
                if date in etf_return_series.index:
                    etf_return = etf_return_series.loc[date]
                    # 处理NaN值，将其视为0%（无变化）
                    if pd.isna(etf_return):
                        etf_return = 0.0
                    etf_return = etf_return * 100
                    weight = etf_info['weight'] * 100
                    contribution = etf_return * etf_info['weight']

                    lines.append(f"├─ {etf_info['name']}: <b>{etf_return:+.2f}%</b> | 占比{weight:.0f}% | 贡献{contribution:+.2f}%")

        # 添加统计信息
        valid_returns = []
        valid_etf_indices = []
        for etf_code in self.etf_codes:
            if etf_code in etf_returns:
                etf_return_series = etf_returns[etf_code]
                # 直接使用传入的date查询对应月份的收益率
                if date in etf_return_series.index:
                    etf_return = etf_return_series.loc[date]
                    # 处理NaN值，将其视为0%（无变化）
                    if pd.isna(etf_return):
                        etf_return = 0.0
                    etf_return = etf_return * 100
                    valid_returns.append(etf_return)
                    valid_etf_indices.append(self.etf_codes.index(etf_code))

        if valid_returns:
            best_return = max(valid_returns)
            worst_return = min(valid_returns)
            best_etf_idx = valid_returns.index(best_return)
            worst_etf_idx = valid_returns.index(worst_return)

            lines.append("")
            lines.append(f"📈 当月最佳: <b>{self.etf_info[self.etf_codes[valid_etf_indices[best_etf_idx]]]['name']} ({best_return:+.2f}%)</b>")
            lines.append(f"📉 当月最差: <b>{self.etf_info[self.etf_codes[valid_etf_indices[worst_etf_idx]]]['name']} ({worst_return:+.2f}%)</b>")

        return "<br>".join(lines)

    def generate_report(self, show_plot=True, show_benchmark=False, benchmark='000300'):
        """
        生成完整的回测报告
        """
        self.print_results()
        self.print_etf_monthly_returns()
        if show_plot:
            self.plot_results(show_benchmark=show_benchmark, benchmark_code=benchmark)
            self.plot_monthly_heatmap()

    def get_monthly_data(self) -> pd.DataFrame:
        """
        获取月度数据用于分析
        """
        if self.portfolio_value is None:
            return pd.DataFrame()

        monthly_data = self.portfolio_value.resample('ME').last()
        monthly_data['monthly_return'] = safe_pct_change(monthly_data['portfolio_value'], fillna=True)
        monthly_data['cumulative_return'] = (monthly_data['portfolio_value'] / monthly_data['portfolio_value'].iloc[0] - 1)

        return monthly_data

    def print_etf_monthly_returns(self, top_n: int = 3):
        """
        打印ETF月度收益率分析

        Args:
            top_n: 显示表现最好和最差的ETF数量
        """
        if self.etf_monthly_values is None:
            print("没有ETF月度数据可显示")
            return

        print("\n" + "="*80)
        print("ETF月度收益率分析")
        print("="*80)

        # 计算每个ETF的月度收益率（基于价格数据）
        etf_monthly_returns = {}
        for etf_code in self.etf_codes:
            try:
                # 获取ETF价格数据
                price_data = get_price_akshare(
                    stock=etf_code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    need_ma=False,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir,
                    force_refresh=self.force_refresh
                )

                if not price_data.empty:
                    # 计算月末价格和收益率，使用安全的方法
                    monthly_prices = price_data.resample('ME').last()
                    returns = safe_pct_change(monthly_prices['close'], fillna=False)  # 不填充NaN，后面会处理
                    etf_monthly_returns[etf_code] = returns
            except Exception as e:
                print(f"获取{etf_code}价格数据时出错: {e}")
                continue

        if not etf_monthly_returns:
            print("没有可用的ETF收益率数据")
            return

        # 创建收益率DataFrame
        returns_df = pd.DataFrame(etf_monthly_returns)

        # 显示每只ETF的统计信息
        print(f"\n各ETF月度收益率统计:")
        print("-" * 80)
        print(f"{'ETF代码':<10} {'平均月收益率':<12} {'最大月收益':<12} {'最小月收益':<12} {'波动率':<10} {'胜率':<8}")
        print("-" * 80)

        for etf_code in self.etf_codes:
            if etf_code in returns_df.columns:
                returns = returns_df[etf_code].dropna()
                avg_return = returns.mean() * 100  # 转换为百分比
                max_return = returns.max() * 100
                min_return = returns.min() * 100
                volatility = returns.std() * 100
                win_rate = (returns > 0).mean() * 100

                print(f"{etf_code:<10} {avg_return:>10.2f}%   {max_return:>10.2f}%   {min_return:>10.2f}%   {volatility:>8.2f}%   {win_rate:>6.1f}%")

        # 显示组合月度收益率（修正：使用考虑定投的月度收益率）
        portfolio_returns = self._calculate_monthly_returns().dropna()

        print("-" * 80)
        print(f"{'投资组合':<10} {portfolio_returns.mean():>10.2f}%   {portfolio_returns.max():>10.2f}%   {portfolio_returns.min():>10.2f}%   {portfolio_returns.std():>8.2f}%   {(portfolio_returns>0).mean()*100:>6.1f}%")
        print("="*80)

        # 显示最好和最差的月份
        print(f"\n表现最好的{top_n}个月份:")
        best_months = portfolio_returns.nlargest(top_n)
        for date, ret in best_months.items():
            etf_contributions = []
            for etf_code in self.etf_codes:
                if etf_code in returns_df.columns:
                    # 查找该ETF在同期的收益率
                    etf_return_series = returns_df[etf_code]
                    if date in etf_return_series.index:
                        etf_ret = etf_return_series.loc[date]
                        if not pd.isna(etf_ret):
                            etf_contributions.append(f"{etf_code}:{etf_ret*100:.1f}%")

            print(f"  {date.strftime('%Y-%m')}: 组合 {ret:.2f}% | {', '.join(etf_contributions)}")

        print(f"\n表现最差的{top_n}个月份:")
        worst_months = portfolio_returns.nsmallest(top_n)
        for date, ret in worst_months.items():
            etf_contributions = []
            for etf_code in self.etf_codes:
                if etf_code in returns_df.columns:
                    # 查找该ETF在同期的收益率
                    etf_return_series = returns_df[etf_code]
                    if date in etf_return_series.index:
                        etf_ret = etf_return_series.loc[date]
                        if not pd.isna(etf_ret):
                            etf_contributions.append(f"{etf_code}:{etf_ret*100:.1f}%")

            print(f"  {date.strftime('%Y-%m')}: 组合 {ret:.2f}% | {', '.join(etf_contributions)}")

    def _prepare_etf_hover_data(self) -> List:
        """
        准备ETF悬停数据用于投资组合价值图

        Returns:
            List: 每个交易日的ETF价值明细数据
        """
        if not hasattr(self, 'etf_daily_values') or self.etf_daily_values is None:
            return [[] for _ in range(len(self.portfolio_value))]

        hover_data = []
        for i in range(len(self.portfolio_value)):
            day_data = []
            total_portfolio_value = self.portfolio_value['portfolio_value'].iloc[i]

            for etf_code in self.etf_codes:
                if etf_code in self.etf_daily_values and i < len(self.etf_daily_values[etf_code]):
                    etf_value = self.etf_daily_values[etf_code][i]
                    initial_etf_value = self.initial_capital * self.weights[self.etf_codes.index(etf_code)]

                    # 如果ETF价值为0，可能是因为该ETF当天没有交易数据
                    # 这种情况下，我们应该显示"无数据"而不是错误的收益率
                    if etf_value == 0 and i > 0:
                        # 检查前一天的值，如果前一天也不是0，说明今天确实没有数据
                        if i-1 < len(self.etf_daily_values[etf_code]) and self.etf_daily_values[etf_code][i-1] > 0:
                            day_data.extend([
                                self.etf_info[etf_code]['name'],
                                "无数据",
                                "无数据",
                                "0.0%"
                            ])
                            continue

                    # 安全计算收益率，避免NaN
                    if initial_etf_value > 0 and etf_value > 0:
                        etf_return = ((etf_value / initial_etf_value) - 1) * 100
                        # 检查是否为有效数值
                        if not np.isfinite(etf_return):
                            etf_return = 0.0
                    else:
                        etf_return = 0.0

                    # 安全计算占比，避免NaN
                    if total_portfolio_value > 0 and etf_value >= 0:
                        etf_percentage = (etf_value / total_portfolio_value) * 100
                        # 检查是否为有效数值
                        if not np.isfinite(etf_percentage):
                            etf_percentage = 0.0
                    else:
                        etf_percentage = 0.0

                    day_data.extend([
                        self.etf_info[etf_code]['name'],
                        f"¥{etf_value:,.0f}" if etf_value >= 0 else "¥0",
                        f"{etf_return:+.2f}%",
                        f"{etf_percentage:.1f}%"
                    ])
                else:
                    day_data.extend([self.etf_info[etf_code]['name'], "¥0", "0.00%", "0.0%"])

            hover_data.append(day_data)

        return hover_data

    def _generate_portfolio_hover_template(self) -> str:
        """
        生成投资组合价值图的悬停模板

        Returns:
            str: 格式化的悬停模板
        """
        template_lines = ["日期: %{x}"]
        template_lines.append("总价值: <b>¥%{y:,.0f}</b>")
        template_lines.append("")
        template_lines.append("📊 <b>ETF明细:</b>")

        for i, etf_code in enumerate(self.etf_codes):
            base_idx = i * 4
            template_lines.append(f"├─ {self.etf_info[etf_code]['name']}: <b>%{{customdata[{base_idx+1}]}}</b> | 收益率 %{{customdata[{base_idx+2}]}} | 占比 %{{customdata[{base_idx+3}]}}")

        return "<br>".join(template_lines) + "<extra></extra>"

    def _prepare_etf_returns_data(self) -> List:
        """
        准备ETF收益率数据用于收益率图

        Returns:
            List: 每个交易日的ETF收益率明细数据
        """
        if not hasattr(self, 'etf_daily_values') or self.etf_daily_values is None:
            return [[] for _ in range(len(self.portfolio_value))]

        hover_data = []
        for i in range(len(self.portfolio_value)):
            day_data = []

            # 计算每个ETF的累计收益率
            for etf_code in self.etf_codes:
                if etf_code in self.etf_daily_values and i < len(self.etf_daily_values[etf_code]):
                    current_value = self.etf_daily_values[etf_code][i]
                    initial_value = self.initial_capital * self.weights[self.etf_codes.index(etf_code)]

                    # 如果ETF价值为0，且不是第一天，说明可能没有数据
                    if current_value == 0 and i > 0:
                        # 检查前一天的值，如果前一天也不是0，说明今天确实没有数据
                        if i-1 < len(self.etf_daily_values[etf_code]) and self.etf_daily_values[etf_code][i-1] > 0:
                            day_data.extend([
                                self.etf_info[etf_code]['name'],
                                "无数据"
                            ])
                            continue

                    # 安全计算收益率，避免NaN
                    if initial_value > 0 and current_value > 0:
                        etf_return = ((current_value / initial_value) - 1) * 100
                        # 检查是否为有效数值
                        if not np.isfinite(etf_return):
                            etf_return = 0.0
                    else:
                        etf_return = 0.0

                    day_data.extend([
                        self.etf_info[etf_code]['name'],
                        f"{etf_return:+.2f}%"
                    ])
                else:
                    day_data.extend([self.etf_info[etf_code]['name'], "0.00%"])

            hover_data.append(day_data)

        return hover_data

    def _generate_returns_hover_template(self) -> str:
        """
        生成收益率图的悬停模板

        Returns:
            str: 格式化的悬停模板
        """
        template_lines = ["日期: %{x}"]
        template_lines.append("累计收益率: <b>%{y:+.2f}%</b>")
        template_lines.append("")
        template_lines.append("📊 <b>ETF收益率明细:</b>")

        for i, etf_code in enumerate(self.etf_codes):
            base_idx = i * 2
            template_lines.append(f"├─ {self.etf_info[etf_code]['name']}: <b>%{{customdata[{base_idx+1}]}}</b>")

        return "<br>".join(template_lines) + "<extra></extra>"
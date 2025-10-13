import pandas as pd
import numpy as np
import datetime
import akshare as ak
from typing import List, Dict, Optional
import os

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

        print(f"成功获取 {stock} 数据，共 {len(result)} 条记录")
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
                 force_refresh: bool = False):
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
            force_refresh: 是否强制刷新缓存数据，默认False
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

        # 数据刷新参数
        self.force_refresh = force_refresh

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
        self.transactions = []  # 交易记录
        self.dca_dates = []  # 定投日期
        self.rebalance_dates = []  # 再平衡日期

        # 回测结果
        self.results = {}

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有ETF的价格数据"""
        if self.force_refresh:
            print("开始获取ETF数据（强制刷新缓存）...")
        else:
            print("开始获取ETF数据...")

        for etf_code in self.etf_codes:
            print(f"获取 {etf_code} 数据...")
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
            print(f"  {etf_code}: {len(data)} 条记录")

        if not self.etf_data:
            raise ValueError("未能获取任何ETF数据")

        return self.etf_data

    def _initial_buy(self, date: pd.Timestamp):
        """
        初始建仓 - 使用开盘价买入

        Args:
            date: 建仓日期
        """
        print(f"\n在 {date.strftime('%Y-%m-%d')} 初始建仓")

        total_value = self.cash

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
                    print(f"  {etf_code}: 在 {date} 后没有交易数据，跳过")
                    continue

            open_price = self.etf_data[etf_code].loc[date, 'open']

            if pd.isna(open_price):
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
                self.transactions.append({
                    'date': date,
                    'type': 'initial_buy',
                    'etf_code': etf_code,
                    'shares': shares,
                    'price': open_price,
                    'amount': cost,
                    'cash_after': self.cash
                })

                print(f"  买入 {etf_code}: {shares}股 @ ¥{open_price:.3f}, 成本: ¥{cost:,.2f}")

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
        print(f"\n定投 {date.strftime('%Y-%m-%d')}: 投入 ¥{self.dca_amount:,.2f}")

        # 添加定投资金
        self.cash += self.dca_amount

        total_dca_value = self.dca_amount

        for i, etf_code in enumerate(self.etf_codes):
            if etf_code not in self.etf_data:
                continue

            # 获取当天的收盘价
            if date not in self.etf_data[etf_code].index:
                print(f"  {etf_code}: 当天无交易数据，跳过")
                continue

            close_price = self.etf_data[etf_code].loc[date, 'close']

            if pd.isna(close_price):
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
                self.transactions.append({
                    'date': date,
                    'type': 'dca_buy',
                    'etf_code': etf_code,
                    'shares': shares,
                    'price': close_price,
                    'amount': cost,
                    'cash_after': self.cash
                })

                print(f"  买入 {etf_code}: {shares:.4f}股 @ ¥{close_price:.3f}, 成本: ¥{cost:,.2f}")

        print(f"  定投后现金: ¥{self.cash:,.2f}")

    def _rebalance_portfolio(self, date: pd.Timestamp):
        """
        执行组合再平衡 - 使用收盘价进行交易

        Args:
            date: 再平衡日期
        """
        print(f"\n再平衡 {date.strftime('%Y-%m-%d')}:")

        # 计算当前组合价值
        total_value = self._calculate_portfolio_value(date)
        if total_value <= 0:
            print("  组合价值为0，跳过再平衡")
            return

        # 计算每个ETF的当前权重和目标权重
        current_weights = {}
        target_values = {}
        total_sell_value = 0
        total_buy_value = 0

        print(f"  当前组合价值: ¥{total_value:,.2f}")

        # 找到当前rebalance周期的开始日期（上一个rebalance日之后的第一天）
        current_rebalance_start = self._get_rebalance_period_start(date)

        # 显示详细的rebalancing信息
        self._print_rebalancing_details(current_rebalance_start, date, total_value)

        # 分析当前持仓和目标调整
        for i, etf_code in enumerate(self.etf_codes):
            if etf_code in self.positions and etf_code in self.etf_data:
                if date in self.etf_data[etf_code].index:
                    current_price = self.etf_data[etf_code].loc[date, 'close']
                    current_value = self.positions[etf_code]['shares'] * current_price
                    current_weight = current_value / total_value
                    target_weight = self.weights[i]
                    target_value = total_value * target_weight

                    current_weights[etf_code] = current_weight
                    target_values[etf_code] = target_value

  
        # 执行再平衡交易
        for i, etf_code in enumerate(self.etf_codes):
            if etf_code not in self.etf_data or date not in self.etf_data[etf_code].index:
                continue

            current_price = self.etf_data[etf_code].loc[date, 'close']
            target_value = target_values.get(etf_code, total_value * self.weights[i])

            if etf_code in self.positions:
                current_value = self.positions[etf_code]['shares'] * current_price
                current_shares = self.positions[etf_code]['shares']

                # 需要调整的金额
                adjust_value = target_value - current_value

                if adjust_value > 0:
                    # 需要买入
                    # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
                    shares_to_buy = adjust_value / (current_price * (1 + self.transaction_cost))
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)

                    if cost <= self.cash + 1:  # 允许小幅误差
                        # 更新持仓
                        new_shares = current_shares + shares_to_buy
                        new_total_cost = self.positions[etf_code]['total_cost'] + cost
                        avg_cost = new_total_cost / new_shares

                        self.positions[etf_code] = {
                            'shares': new_shares,
                            'avg_cost': avg_cost,
                            'total_cost': new_total_cost
                        }
                        self.cash -= cost
                        total_buy_value += cost

                        # 记录交易
                        self.transactions.append({
                            'date': date,
                            'type': 'rebalance_buy',
                            'etf_code': etf_code,
                            'shares': shares_to_buy,
                            'price': current_price,
                            'amount': cost,
                            'cash_after': self.cash
                        })

                        print(f"  买入 {etf_code}: {shares_to_buy:.4f}股 @ ¥{current_price:.3f}, 成本: ¥{cost:,.2f}")

                elif adjust_value < 0:
                    # 需要卖出
                    shares_to_sell = abs(adjust_value) / current_price
                    shares_to_sell = min(shares_to_sell, current_shares)  # 不能卖出超过持有的数量

                    if shares_to_sell > 0:
                        sell_proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                        remaining_shares = current_shares - shares_to_sell

                        if remaining_shares > 0:
                            # 更新持仓
                            remaining_cost_ratio = remaining_shares / current_shares
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
                        total_sell_value += sell_proceeds

                        # 记录交易
                        self.transactions.append({
                            'date': date,
                            'type': 'rebalance_sell',
                            'etf_code': etf_code,
                            'shares': shares_to_sell,
                            'price': current_price,
                            'amount': sell_proceeds,
                            'cash_after': self.cash
                        })

                        print(f"  卖出 {etf_code}: {shares_to_sell:.4f}股 @ ¥{current_price:.3f}, 收入: ¥{sell_proceeds:,.2f}")

            else:
                # 新建持仓
                target_value = total_value * self.weights[i]
                if target_value > 0 and self.cash > 0:
                    # 正确的交易成本计算：目标投资额需要同时覆盖股票价值和交易成本
                    # 但需要考虑可用现金的限制
                    max_affordable_value = self.cash / (1 + self.transaction_cost)
                    actual_target_value = min(target_value, max_affordable_value)
                    shares_to_buy = actual_target_value / (current_price * (1 + self.transaction_cost))
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)

                    if cost <= self.cash:
                        self.positions[etf_code] = {
                            'shares': shares_to_buy,
                            'avg_cost': current_price,
                            'total_cost': cost
                        }
                        self.cash -= cost
                        total_buy_value += cost

                        # 记录交易
                        self.transactions.append({
                            'date': date,
                            'type': 'rebalance_buy',
                            'etf_code': etf_code,
                            'shares': shares_to_buy,
                            'price': current_price,
                            'amount': cost,
                            'cash_after': self.cash
                        })

                        print(f"  新建持仓 {etf_code}: {shares_to_buy:.4f}股 @ ¥{current_price:.3f}, 成本: ¥{cost:,.2f}")

        print(f"  再平衡后现金: ¥{self.cash:,.2f}")
        print(f"  总买入: ¥{total_buy_value:,.2f}, 总卖出: ¥{total_sell_value:,.2f}")

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

        return total_value

    def run_backtest(self):
        """
        运行回测
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

        print(f"\n开始回测，共 {len(trading_dates)} 个交易日")

        # 计算定投日期
        self.dca_dates = self._get_dca_dates(trading_dates)
        if self.enable_dca:
            print(f"定投计划: {self.dca_freq}, 共 {len(self.dca_dates)} 次定投")
            for d in self.dca_dates[:5]:  # 只显示前5个
                print(f"  {d.strftime('%Y-%m-%d')}")
            if len(self.dca_dates) > 5:
                print(f"  ... 还有 {len(self.dca_dates) - 5} 次定投")

        # 计算再平衡日期
        self.rebalance_dates = self._get_rebalance_dates(trading_dates)
        if self.enable_rebalancing:
            print(f"再平衡计划: {self.rebalance_freq}, 阈值 {self.rebalance_threshold:.1%}, 共 {len(self.rebalance_dates)} 次再平衡")
            for d in self.rebalance_dates[:5]:  # 只显示前5个
                print(f"  {d.strftime('%Y-%m-%d')}")
            if len(self.rebalance_dates) > 5:
                print(f"  ... 还有 {len(self.rebalance_dates) - 5} 次再平衡")

        # 第一天初始建仓
        if trading_dates:
            self._initial_buy(trading_dates[0])

        # 逐日更新
        for i, date in enumerate(trading_dates):
            # 检查是否是定投日
            if date in self.dca_dates:
                self._dca_buy(date)

            # 检查是否是再平衡日
            if date in self.rebalance_dates:
                self._rebalance_portfolio(date)

            # 计算当日组合价值
            portfolio_value = self._calculate_portfolio_value(date)

            # 记录
            self.daily_dates.append(date)
            self.daily_values.append(portfolio_value)

            # # 每20天打印一次进度
            # if i % 20 == 0:
            #     print(f"  处理进度: {date.strftime('%Y-%m-%d')}, 组合价值: ¥{portfolio_value:,.2f}")

        # 计算回测结果
        self._calculate_results()

        print("\n回测完成!")
        self.print_results()

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

        # 计算收益率
        returns = value_series.pct_change().fillna(0)

        # 总收益率（基于总投入）
        total_return = (value_series.iloc[-1] / total_investment - 1) * 100

        # 年化收益率
        days = (value_series.index[-1] - value_series.index[0]).days
        years = days / 365.25
        if years > 0:
            annual_return = (value_series.iloc[-1] / total_investment) ** (1/years) - 1
            annual_return_pct = annual_return * 100
        else:
            annual_return_pct = 0

        # 最大回撤
        rolling_max = value_series.expanding().max()
        drawdown = (value_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # 年化波动率
        volatility = returns.std() * np.sqrt(252) * 100

        # 夏普比率（假设无风险利率为3%）
        rf_rate = 0.03
        sharpe_ratio = (annual_return - rf_rate) / (volatility / 100) if volatility > 0 else 0

        self.results = {
            'initial_capital': self.initial_capital,
            'total_investment': total_investment,
            'final_value': value_series.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return_pct,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
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
        print(f"年化波动率: {self.results['volatility']:.2f}%")
        print(f"夏普比率: {self.results['sharpe_ratio']:.3f}")
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

    def get_results(self) -> Dict:
        """获取回测结果"""
        return self.results
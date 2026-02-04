# AABacktest - 资产配置策略回测系统

一个专业的ETF资产配置策略回测系统，支持多种再平衡频率、定投策略（DCA）、交易成本计算、智能日期处理、数据缓存功能和交互式图表可视化。

## ✨ 主要特性

- 🔄 **多种再平衡策略**：支持年度、季度、月度再平衡，以及阈值触发再平衡
- 💰 **定投策略支持**：支持定期定额投资（DCA），可自定义定投频率和金额
- 📊 **交互式图表**：基于Plotly的完整可视化仪表板，包含5个子图分析
- 💾 **智能数据缓存**：本地缓存akshare数据，提高稳定性，支持离线回测
- 🔧 **详细程度控制**：支持简单模式和详细模式的交易信息显示
- 📈 **完整性能指标**：总收益率、年化收益率、最大回撤、夏普比率、回补周期等
- 🎯 **精确交易成本**：分别计算买入和卖出成本，支持自定义交易成本率
- 📅 **智能日期处理**：自动识别交易日，避免节假日跳过重要操作
- 🇨🇳 **A股ETF支持**：使用akshare获取国内ETF、股票和指数数据

## 📦 安装

### 使用uv（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd AABacktest

# 安装依赖
uv sync

# 运行示例
uv run python example_usage.py
```

### 使用pip

```bash
# 安装依赖
pip install pandas numpy akshare plotly

# 运行示例
python example_usage.py
```

## 🚀 快速开始

### 基础使用示例

```python
from portfolio_backtester import PortfolioBacktester

# 创建回测实例
backtester = PortfolioBacktester(
    etf_codes=['511010', '510880', '518880', '513100'],  # 国债、红利、黄金、纳斯达克
    weights=[0.3, 0.4, 0.1, 0.2],                        # 对应权重
    enable_rebalancing=True,
    rebalance_freq='yearly',
    start_date='2015-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    transaction_cost=0.001,                               # 0.1%交易成本
    rebalance_threshold=0.01,                             # 1%阈值触发再平衡（新功能）
    show_daily_logs=True,                                 # 显示每日交易日志（新功能）
    verbose_trading=False                                 # 简单模式
)

# 运行回测
backtester.run_backtest()

# 生成完整报告（包含交互式图表）
backtester.generate_report(show_plot=True)
```

### 定投策略示例

```python
# 定投策略示例 - 每年定投
backtester_dca = PortfolioBacktester(
    etf_codes=['511010', '510880', '518880', '513100'],
    weights=[0.25, 0.25, 0.25, 0.25],
    enable_rebalancing=True,
    rebalance_freq='yearly',
    enable_dca=True,                                       # 启用定投
    dca_amount=10000,                                      # 每次定投1万元
    dca_freq='yearly',                                     # 年度定投
    start_date='2015-01-01',
    end_date='2023-12-31',
    initial_capital=50000,                                 # 初始资金5万元
    verbose_trading=False
)

backtester_dca.run_backtest()
backtester_dca.generate_report()
```

### 详细模式示例

```python
# 详细模式 - 显示所有交易细节
backtester_verbose = PortfolioBacktester(
    etf_codes=['511010', '510880', '518880', '513100'],
    weights=[0.3, 0.4, 0.1, 0.2],
    enable_rebalancing=True,
    rebalance_freq='quarterly',
    enable_dca=True,
    dca_amount=5000,
    dca_freq='quarterly',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=50000,
    initial_capital=50000,
    show_daily_logs=True,                                 # 显示日志
    verbose_trading=True                                  # 详细模式
)

backtester_verbose.run_backtest()
backtester_verbose.generate_report()
```

## 📋 参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `etf_codes` | `List[str]` | 必填 | ETF代码列表 |
| `weights` | `List[float]` | 必填 | 对应权重列表（总和必须为1） |
| `start_date` | `str` | 必填 | 回测开始日期 (格式: 'YYYY-MM-DD') |
| `end_date` | `str` | 必填 | 回测结束日期 (格式: 'YYYY-MM-DD') |
| `initial_capital` | `float` | 100000 | 初始资金 |
| `transaction_cost` | `float` | 0.000 | 交易成本比例（0.001 = 0.1%） |

### 再平衡参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_rebalancing` | `bool` | False | 是否启用再平衡 |
| `rebalance_freq` | `str` | 'quarterly' | 再平衡频率：'monthly', 'quarterly', 'yearly' |
| `rebalance_threshold` | `float` | 0.0 | 再平衡触发阈值（权重偏离超过此值时强制触发，如0.01=1%） |

### 定投参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_dca` | `bool` | False | 是否启用定投 |
| `dca_amount` | `float` | 10000 | 定投金额 |
| `dca_freq` | `str` | 'monthly' | 定投频率：'monthly', 'yearly' |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `risk_free_rate` | `float` | 0.02 | 无风险利率（用于计算夏普比率） |
| `force_refresh` | `bool` | False | 是否强制刷新缓存数据 |
| `verbose_trading` | `bool` | False | 是否显示详细的交易信息（为False时显示简略汇总） |
| `show_daily_logs` | `bool` | True | 是否显示每日交易/再平衡日志开关 |

## 📊 支持的ETF代码

### 常用A股ETF

| 代码 | 名称 | 类别 |
|------|------|------|
| `511010` | 国债ETF | 债券 |
| `510880` | 红利ETF | 股票 |
| `518880` | 黄金ETF | 商品 |
| `513100` | 纳斯达克100ETF | 海外股票 |
| `159915` | 创业板ETF | 股票 |
| `510300` | 沪深300ETF | 股票 |
| `159919` | 沪深300ETF | 股票 |
| `512100` | 中证1000ETF | 股票 |
| `516160` | 新能源ETF | 行业 |
| `515050` | 5GETF | 行业 |

### 数据获取策略

系统会按以下顺序尝试获取数据：
1. 股票数据（ak.stock_zh_a_hist）
2. ETF数据（ak.fund_etf_hist_em）
3. 指数数据（ak.index_zh_a_hist）

## 📈 回测结果指标

### 基础指标

- **初始资金**：策略开始时的资金
- **总投入**：初始资金 + 定投总额
- **最终价值**：回测结束时的组合价值
- **总收益率**：基于总投入的收益率
- **年化收益率**：复合年增长率

### 风险指标

- **最大回撤**：从最高点到最低点的最大跌幅
- **最大回撤回补周期**：从最大回撤到恢复至前期高点所需时间
- **年化波动率**：收益率的标准差（年化）
- **夏普比率**：风险调整后收益

### 交易统计

- **再平衡次数**：执行再平衡的次数
- **定投次数**：执行定投的次数
- **交易天数**：回测期间的总交易日数

## 🎨 可视化图表

系统自动生成包含5个子图的交互式仪表板：

1. **账户资金变化**：显示总资产价值随时间的变化，标记再平衡和定投日期
2. **累计收益率**：显示投资收益率的时间序列
3. **回撤分析**：显示组合回撤情况和最大回撤点
4. **月度收益率热力图**：以热力图形式显示每月收益率
5. **年度收益率**：柱状图显示年度收益表现

### 图表交互功能

- 🖱️ **悬停显示**：鼠标悬停显示详细数值
- 🔍 **缩放功能**：支持图表缩放和平移
- 📊 **动态标记**：自动标记再平衡和定投日期
- 🎨 **专业配色**：使用专业的金融图表配色方案

## 💾 数据缓存机制

### 缓存特性

- **自动缓存**：成功获取的数据自动保存到本地
- **智能匹配**：自动查找包含所需时间范围的现有缓存文件
- **离线支持**：有缓存时可在无网络环境下运行
- **强制刷新**：支持强制重新获取最新数据

### 缓存管理

```python
# 使用缓存（默认）
backtester = PortfolioBacktester(
    etf_codes=['511010', '510880'],
    weights=[0.5, 0.5],
    force_refresh=False  # 使用缓存
)

# 强制刷新缓存
backtester_new = PortfolioBacktester(
    etf_codes=['511010', '510880'],
    weights=[0.5, 0.5],
    force_refresh=True   # 强制刷新
)
```

## 🔧 高级功能

### 阈值触发再平衡

```python
# 当权重偏离超过5%时触发再平衡
backtester = PortfolioBacktester(
    etf_codes=['511010', '510880', '518880', '513100'],
    weights=[0.25, 0.25, 0.25, 0.25],
    enable_rebalancing=True,
    rebalance_freq='yearly',          # 定期再平衡
    rebalance_threshold=0.05,         # 阈值触发：偏离5%强制再平衡
    start_date='2018-01-01',
    end_date='2023-12-31'
)
```

### 详细程度控制

```python
# 简单模式 - 只显示汇总信息
backtester_simple = PortfolioBacktester(
    # ... 其他参数
    verbose_trading=False
)

# 详细模式 - 显示每笔交易详情
backtester_detail = PortfolioBacktester(
    # ... 其他参数
    verbose_trading=True
)
```

### 日志显示控制

```python
# 静默模式 - 只显示最终报告，不刷屏
backtester_silent = PortfolioBacktester(
    # ... 其他参数
    show_daily_logs=False
)
```

## 📋 使用示例

### 运行内置示例

```bash
# 基础示例（简单模式）
uv run python example_usage.py

# 详细模式示例
uv run python test_verbose.py
```

### 自定义策略对比

```python
# 对比不同再平衡频率
frequencies = ['yearly', 'quarterly', 'monthly']
results = {}

for freq in frequencies:
    backtester = PortfolioBacktester(
        etf_codes=['511010', '510880', '518880', '513100'],
        weights=[0.25, 0.25, 0.25, 0.25],
        enable_rebalancing=True,
        rebalance_freq=freq,
        start_date='2015-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        verbose_trading=False
    )
    backtester.run_backtest()
    results[freq] = backtester.get_results()

# 对比结果
for freq, result in results.items():
    print(f"{freq}: 总收益率 {result['total_return']:.2f}%, 夏普比率 {result['sharpe_ratio']:.3f}")
```

## ⚠️ 注意事项

1. **数据依赖**：需要网络连接获取akshare数据，首次运行会较慢
2. **交易日历**：遵循中国A股交易日历，自动处理节假日
3. **碎股交易**：回测中支持碎股交易，与实际A股交易规则存在差异
4. **历史表现**：历史回测结果不代表未来表现，仅供参考
5. **数据质量**：建议使用流动性好的主流ETF，避免数据缺失

## 🏗️ 项目结构

```
AABacktest/
├── portfolio_backtester.py  # 核心回测系统
├── example_usage.py         # 基础使用示例
├── test_verbose.py          # 详细模式示例
├── pyproject.toml          # 项目配置
├── README.md               # 项目文档
└── data/                   # 数据缓存目录
    └── *.csv              # 缓存的ETF数据文件
```

## 📝 更新日志

### v2.1.0 (最新)
- ✅ 新增**阈值再平衡**功能：即使未到定期平衡日，当权重偏离超过设定阈值（`rebalance_threshold`）时也会强制平衡
- ✅ 优化**交易日志**：
  - 简单模式日志现在显示具体的买卖标的和金额（如 `卖出[510300(¥5k)]`）
  - 新增 `show_daily_logs` 开关，可选择关闭每日刷屏日志，仅看结果
- ✅ 更新文档和示例代码

### v2.0.0
- ✅ 新增定投策略支持（DCA）
- ✅ 新增详细程度控制（verbose_trading参数）
- ✅ 优化再平衡逻辑，采用"先平衡后定投"策略
- ✅ 完善交易成本计算，避免重复扣除
- ✅ 改进数据处理，修复ETF缺失数据的显示问题
- ✅ 删除plotly条件检查，简化依赖管理
- ✅ 增强可视化图表，添加5个子图的综合仪表板

### v1.0.0
- ✅ 基础再平衡策略实现
- ✅ 数据缓存机制
- ✅ 交互式图表支持
- ✅ 完整的回测指标计算

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。
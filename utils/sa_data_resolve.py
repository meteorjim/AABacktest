import requests
import pandas as pd
from datetime import datetime


def fetch_seeking_alpha_to_csv(
    ticker: str = "qqq",
    start_date: str = None,
    end_date: str = None,
    output: bool = False,
    cookies_list: list = None
) -> pd.DataFrame:
    """
    从 Seeking Alpha 获取历史价格数据并保存为 CSV
    
    Args:
        ticker: 股票代码，默认 "qqq"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"
        output_path: 输出文件路径
        cookies_list: cookie 列表
    
    Returns:
        DataFrame
    """
    
    # 构建查询参数
    params = {
        "filter[ticker][slug]": ticker,
        "sort": "as_of_date"
    }
    
    # 添加日期过滤（转换为 API 需要的格式）
    if start_date:
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        params["filter[as_of_date][gte]"] = dt.strftime("%a %b %d %Y")
    if end_date:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        params["filter[as_of_date][lte]"] = dt.strftime("%a %b %d %Y")
    
    # 转换 cookies
    cookies = {}
    if cookies_list:
        for cookie in cookies_list:
            cookies[cookie["name"]] = cookie["value"]
    
    # 请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    # 发送请求
    response = requests.get(
        "https://seekingalpha.com/api/v3/historical_prices",
        params=params,
        cookies=cookies,
        headers=headers
    )
    response.raise_for_status()
    data = response.json()
    
    # 解析数据
    records = []
    for item in data.get("data", []):
        attrs = item.get("attributes", {})
        records.append({
            "date": attrs.get("as_of_date"),
            "open": attrs.get("open"),
            "high": attrs.get("high"),
            "low": attrs.get("low"),
            "close": attrs.get("close"),
            "volume": attrs.get("volume"),
        })
    
    # 创建 DataFrame
    df = pd.DataFrame(records)
    
    if df.empty:
        print("没有获取到数据")
        return df
    
    # 反转顺序（从旧到新）
    df = df[::-1].reset_index(drop=True)
    
    # 格式化日期
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # 计算涨跌幅
    df["cr"] = df["close"].pct_change()
    
    # 计算 Change %
    df["Change %"] = (df["cr"] * 100).round(2).astype(str) + "%"
    
    # 处理 NaN 和特殊情况
    df.loc[0, "Change %"] = "0.00%"  # 第一天没有涨跌
    df["Change %"] = df["Change %"].replace("nan%", "0.00%")
    
    # 格式化 volume（添加千分位）
    df["volume"] = df["volume"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )
    
    # 调整列顺序
    df = df[["date", "open", "high", "low", "close", "Change %", "volume", "cr"]]
    start_date = df["date"].iloc[0]
    # 设置 date 为索引
    df = df.set_index("date")
    df = df[::-1]
    # 保存 CSV
    if output is True:
        output = f"data/{ticker.upper()}_{start_date.replace("-","")}_{end_date.replace("-","")}.csv"
    
    df.to_csv(output)
    print(f"数据已保存到: {output}")
    print(f"共获取 {len(df)} 条数据")
    
    return df


# ==================== Cookies 配置 ====================
# 建议把 cookies 保存到单独的文件或环境变量中
COOKIES = []


# ==================== 使用示例 ====================
if __name__ == "__main__":
    nowaday = datetime.now().strftime("%Y-%m-%d")
    # 获取数据并保存为 CSV
    df = fetch_seeking_alpha_to_csv(
        # ticker="XAUUSD:CUR",
        ticker="USO",
        start_date="2005-01-01",
        end_date=nowaday,
        output=True,
        cookies_list=COOKIES
    )
    
    # 查看结果
    print("\n数据预览:")
    print(df.head())

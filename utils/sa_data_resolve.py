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
COOKIES = [
    {"name": "machine_cookie", "value": "fut6k0w81z1770628833"},
    {"name": "_pxvid", "value": "9895aff5-0598-11f1-a7db-790042ff016f"},
    {"name": "sa-user-id", "value": "s%253A0-72836011-761e-4716-5a68-fa2200345b39.W9jbnO1p3XPzx3sHiKji1PxU%252B7R9M3FgfST8%252F%252BaLK20"},
    {"name": "sa-user-id-v3", "value": "s%253AAQAKIJoEwKqmb5CaNAG2rL7umCfwHC12UoVAuYCvT9zsqyG8EJIFGAIgwoyzwwY6BDPk3LlCBDZMlCw.4zOcgwDvGABG3YqIZpCQA1hf6EANF2w4IXTaDhHeSus"},
    {"name": "user_id", "value": "64133903"},
    {"name": "sapu", "value": "101"},
    {"name": "user_remember_token", "value": "8541f2a9328126c3bf13895952689a587df9b077"},
    {"name": "sailthru_hid", "value": "bc9142dbae748dfbd91b7c215a43b4e96989a7060d8b92a64601ff9afca81d6159f96e63d104ec1000cddbcf"},
    {"name": "_px3", "value": "8e237db9a6372a9a9ce56898c034173953420ebc3d1ed8d2604e91f72a41d36b:2kUVB/THV4h2KlK4q/uv1j95fr/A0SdN5nDoFDWFpff6yELMY6tjpp/2JbW6rPyU99j/BxvsAy4BCZp/ES/FMA==:1000:tKTuQ1c51BhHhhWU1+i0MdkkoN/CNTPr63ZR+QQCybn/Xi1HBu5LmtOuHlLyQh0rIgCE5GQaujb7LRZtwL1p+mw1cnPECPEYZ60zFcW4og7fNJD9v8HmDuVAlCM8uBUln41eZzITFo4aFRdzDXdykD0ERkKGHSDqE2FE6FkXIV0RuQSR/9Ygl13v/HeyEzMNID1UlHybK88o+viyI75OZD65LDqTVx0AgVKktz2OgkjryrNa7letZhkyA2qv8z8Qb8ZR/DdODFOMgvtZlUkLsJ+Aiemo0nVVEA2I9gg9hrv/1giiKUSzWNNNO15ke0ikagYGCljGHTzZd9RKKYLhUMEA2DwaLwwdpYK22ctVgF9noBxvuvez6OYziXyZIPW7Wc5y+z3L5q4il3jE2MuyU03bi2NuGMCcnS0t/4SlAVMhRqipvosFVWsrQQrBg87kqxKQeTvE6L+xap7pf4rtWM+O4eNMnWgYabmlsY9aS6f+6wi1lM2n2kzPtQcLmwLqOi/2dZGVLwiFzDxk+Oe2QaK4Fi2MBVRVGe2qwrZ5hGCjVD+P28/V3uISF03pf5aL53Ugz4CsLuwcLDHnndYUXg=="},
    {"name": "user_cookie_key", "value": "9bb7hi"},
    {"name": "gk_user_access", "value": "1**1771896938"},
    {"name": "gk_user_access_sign", "value": "0b6c6077ee0f74d877d68c01398876decaadc192"},
    {"name": "_ga", "value": "GA1.1.1514771253.1770628857"},
    {"name": "_ga_KGRFF2R2C5", "value": "GS2.1.s1771896940$o2$g0$t1771896940$j60$l0$h0"},
]


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 获取数据并保存为 CSV
    df = fetch_seeking_alpha_to_csv(
        ticker="QQQ",
        start_date="2005-01-01",
        end_date="2026-02-24",
        output=True,
        cookies_list=COOKIES
    )
    
    # 查看结果
    print("\n数据预览:")
    print(df.head())

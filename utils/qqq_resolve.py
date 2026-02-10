import pandas as pd 
df = pd.read_csv("qqqraw.csv")
df=df[::-1]
df['Date'] = pd.to_datetime(df['Date'], format='%b. %d, %Y')
df["Date"] = df['Date'].dt.strftime('%Y-%m-%d')
df.rename(columns={"Date": "date", "Open": "open","High": "high","Low": "low","Close": "close","Volume": "volume"}, inplace=True)
df["cr"]=df["close"].pct_change()
df.set_index("date").to_csv("index_data/QQQ_20060209_20260206.csv")

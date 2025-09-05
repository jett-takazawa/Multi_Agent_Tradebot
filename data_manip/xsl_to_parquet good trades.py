import pandas as pd

excel_path   = "data/rawdata/Goodtrades.xlsx"
parquet_path = "data/recommendations/goodtrades_2025-07-24.parquet"

df = pd.read_excel(excel_path, usecols=["trade_id","timestamp", "symbol", "trend", "pattern", "entry", "stop","exit", "Odds_Score", "strength", "time", "freshness", "trend_alignment","News_Volatility",])
# ensure trade_id is all strings
df['trade_id'] = df['trade_id'].astype(str)

# now this will succeed
df.to_parquet(parquet_path, index=False)

print("Parquet saved ➜", parquet_path)
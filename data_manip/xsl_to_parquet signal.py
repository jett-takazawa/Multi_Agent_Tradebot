import pandas as pd

excel_path   = "data/rawdata/Signal Train.xlsx"
parquet_path = "data/recommendations/signal_2025-07-24.parquet"

df = pd.read_excel(excel_path, usecols=["timestamp", "symbol", "trend", "pattern", "entry", "stop","exit", "Odds_Score", "strength", "time", "freshness", "trend_alignment",])


# now this will succeed
df.to_parquet(parquet_path, index=False)

print("Parquet saved ➜", parquet_path)
import pandas as pd

excel_path   = "data/rawdata/Executive_Train.xlsx"
parquet_path = "data/recommendations/execDecision_2025-07-24.parquet"

df = pd.read_excel(excel_path, usecols=["trade_id","exec_output"])

# ensure trade_id is all strings
df['trade_id'] = df['trade_id'].astype(str)

# now this will succeed
df.to_parquet(parquet_path, index=False)

print("Parquet saved ➜", parquet_path)
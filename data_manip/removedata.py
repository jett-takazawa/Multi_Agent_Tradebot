
#TO REMOVE DATA WITH DEFF = X

import pandas as pd

# 1. Read your Parquet file
df = pd.read_parquet("data/signals/signals_2025-07-20.parquet")

# 2. Drop the row(s) you don’t want.
#    Example: remove all rows where `id == 123`
df_filtered = df[df["reason"] != "LLM_ERROR"]

# 3. Write back out to a new Parquet
df_filtered.to_parquet("data/signals/signals_2025-07-20.parquet", index=False)



#TO REMOVE THINGS BEFORE A TIME WITH TIMESTAMP

"""
import pandas as pd

# 1. Read your Parquet file
df = pd.read_parquet("data/signals/signals_2025-07-16.parquet")

# 2. Ensure your timestamp column is a datetime dtype
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 3. Define your cutoff instant
cutoff = pd.to_datetime("2025-07-15T01:56:08.942741")

# 4. Keep only rows at or after the cutoff
df_filtered = df[df["timestamp"] >= cutoff]

# 5. (Optional) drop rows by other criteria, e.g. no_trade
# df_filtered = df_filtered[df_filtered["decision"] != "NO_TRADE"]

# 6. Write back out to Parquet
df_filtered.to_parquet("data/signals/signals_2025-07-14.parquet", index=False)"""
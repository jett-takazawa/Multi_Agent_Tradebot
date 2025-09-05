import pandas as pd

fp = "data/recommendations/recommend_2025-07-02.parquet"
df = pd.read_parquet(fp)

print("\nShape  :", df.shape)
print("Columns:", list(df.columns))
print("Head   :\n", df.head())
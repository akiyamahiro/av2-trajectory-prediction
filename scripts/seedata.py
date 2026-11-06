from pathlib import Path
import pandas as pd

path = Path(
    "data/raw/train/000fd3cb-1fa5-42d3-b532-533474c70c13/"
    "scenario_000fd3cb-1fa5-42d3-b532-533474c70c13.parquet"
)

df = pd.read_parquet(path)
print(type(df))

# 行数、列数
print(df.shape)

# 所有列名
print(df.columns.tolist())

# 前5行
print(df.head())

# 每列的数据类型
print(df.dtypes)

# 各列非空数量、类型及内存信息

df.info()
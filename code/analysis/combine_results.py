import pandas as pd
from pathlib import Path

out_dir = "../../out/experiments"

exp_directory = Path("../../out/experiments")
user_files = list(exp_directory.glob("*/*/users_predictions.csv"))
content_files = list(exp_directory.glob("*/*/content_predictions.csv"))

#obtengo los dataframes
user_dfs = []
for f in user_files:
    try:
        df = pd.read_csv(f)
        df.reset_index(inplace=True)
    except pd.errors.EmptyDataError:
        continue

    user_dfs.append(df)

content_dfs = []
for f in content_files:
    try:
        df = pd.read_csv(f)
        df.reset_index(inplace=True)
    except pd.errors.EmptyDataError:
        continue

    content_dfs.append(df)



#combino y guardo
if user_dfs:
    cols = [c for c in user_dfs[0].columns if all(c in df.columns for df in user_dfs)]
    combined = pd.concat([df[cols] for df in user_dfs]).drop("index",axis=1)
    combined.to_csv(f"{out_dir}/users_predictions.csv",index=False)

if content_dfs:
    cols = [c for c in content_dfs[0].columns if all(c in df.columns for df in content_dfs)]
    combined = pd.concat([df[cols] for df in content_dfs],ignore_index=True).drop("index",axis=1)
    combined.to_csv(f"{out_dir}/content_predictions.csv",index=False)

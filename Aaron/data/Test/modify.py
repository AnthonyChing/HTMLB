
import numpy as np
import scipy as sp
import pandas as pd

stage_1 = pd.read_csv("stage_1_filled.csv")
stage_2 = pd.read_csv("stage_2_filled.csv")

data = [stage_1, stage_2]

for i in [0, 1]:
    data[i].insert(data[i].columns.get_loc("away_pitcher_rest"), "season", data[i]['year'])
    data[i] = data[i].drop('year', axis=1)
    data[i].to_csv(f"stage_{i + 1}.csv", index_label="id")
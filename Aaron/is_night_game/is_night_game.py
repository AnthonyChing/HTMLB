import pandas as pd
import numpy as np
import os

stage_1 = pd.read_csv(os.getcwd() + "/../../stage 1/same_season_test_data.csv")
stage_2 = pd.read_csv(os.getcwd() + "/../../stage 2/2024_test_data.csv")

stage_1 = stage_1.dropna(subset=['is_night_game'])
stage_2 = stage_2.dropna(subset=['is_night_game'])

stage_1_label = stage_1[['is_night_game']]
stage_2_label = stage_2[['is_night_game']]

stage_1_label.reset_index(inplace=True, drop=True)
stage_2_label.reset_index(inplace=True, drop=True)

stage_1_label.to_csv("isnightgame_stage_1_label.csv", index_label='id', index=True)
stage_2_label.to_csv("isnightgame_stage_2_label.csv", index_label='id', index=True)
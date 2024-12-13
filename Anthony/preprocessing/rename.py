import pandas as pd
# Load original and new CSV files
# Convert training data
df = pd.read_csv('updated_original.csv')
name_df = pd.read_csv('match_name.csv')

name = ["KFH","JBM","VQC","XFB","QDH","DPS","YHA","PDF","ECN","STC","VJV","RAV","UPV","FBW","GUT","HAN","GLO","RKN","GKO","SAJ","KJP","PJT","QPO","ZQF","BPH","MOO","RLJ","HXK","MZG","JEM"]
score = [0] * 30

for index, row in df.iterrows():
    if row['home_team_win'] == "True":
        i = name.index(df.at[index, 'home_team_abbr'])
        score[i] += 1
    else:
        i = name.index(df.at[index, 'away_team_abbr'])
        score[i] += 1
        
print(score)

name_df['wins'] = score
name_df = name_df.sort_values(by='wins')

new_index = [i for i in range(30)]
name_df['new_index'] = new_index
# name_df.to_csv('wins.csv', index=False)

rank = ["ECN","RKN","UPV","ZQF","JBM","PDF","RLJ","JEM","VJV","SAJ","GUT","BPH","HAN","YHA","PJT","RAV","MOO","KJP","XFB","VQC","HXK","STC","GKO","QDH","QPO","KFH","MZG","GLO","FBW","DPS"]

df['home_team_index'] = [0.0] * 11067
df['away_team_index'] = [0.0] * 11067
for index, row in df.iterrows():
    df.at[index, 'home_team_index'] = rank.index(row['home_team_abbr']) / 30.0
    df.at[index, 'away_team_index'] = rank.index(row['away_team_abbr']) / 30.0

df.to_csv('reindex.csv', index=False)

# Convert testing data
df = pd.read_csv('updated_test.csv')
df['home_team_index'] = [0.0] * 6185
df['away_team_index'] = [0.0] * 6185
for index, row in df.iterrows():
    df.at[index, 'home_team_index'] = rank.index(row['home_team_abbr']) / 30.0
    df.at[index, 'away_team_index'] = rank.index(row['away_team_abbr']) / 30.0

df.to_csv('reindex_test.csv', index=False)

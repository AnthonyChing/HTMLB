import pandas as pd
df = pd.read_csv('reindex.csv')
df = df.drop(columns=['id', 'home_team_abbr', 'away_team_abbr', 'date', 'is_night_game', 'home_pitcher', 'away_pitcher', 'season', 'home_team_season', 'away_team_season'])

df.to_csv('train.csv', index=False)

df = pd.read_csv('reindex_test.csv')
df = df.drop(columns=['id', 'home_team_abbr', 'away_team_abbr', 'is_night_game', 'home_pitcher', 'away_pitcher', 'season', 'home_team_season', 'away_team_season'])

df.to_csv('test.csv', index=False)
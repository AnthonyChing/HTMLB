import pandas as pd
import sys

file = sys.argv[1]
ans = sys.argv[2]

predictions = pd.read_csv(file)
answers = pd.read_csv(ans)

correct = (predictions['home_team_win'] == answers['home_team_win']).sum()

total = len(predictions)
accuracy = (correct / total) * 100

print(f"Accuracy: {accuracy:.3f}%")

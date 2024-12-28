import pandas as pd
import sys

file = sys.argv[1]
ans = sys.argv[2]

predictions = pd.read_csv(file)
answers = pd.read_csv(ans)

correct = (predictions['is_night_game'] == answers['is_night_game']).sum()

total = len(predictions)
accuracy = (correct / total) * 100

print(f"Accuracy: {accuracy:.3f}%")

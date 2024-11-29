import random
f = open("conflip.csv", "w")
lines = ["id,home_team_win\n"]
for i in range(2428):
    seq = ["True", "False"]
    lines.append(f"{i},{random.choice(seq=seq)}\n")
f.writelines(lines)
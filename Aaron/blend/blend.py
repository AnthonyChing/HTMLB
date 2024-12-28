import pandas as pd
import numpy as np
import os
import sys

#  Blends four files, break ties with True
data = []
for i in range(4):
    data.append(pd.read_csv(sys.argv[i + 1]))

all = pd.concat(data, axis=1)

def blend_votes(row):
    true_count = row.value_counts().get(True, 0)
    false_count = row.value_counts().get(False, 0)
    # Default to TRUE in case of a tie
    return True if true_count >= false_count else False

# Apply the blending logic row-wise
blended = all.apply(blend_votes, axis=1)

# Create a new DataFrame with the blended results
blended_df = pd.DataFrame({data[0].columns[1]: blended})

# Save the result to a new CSV
blended_df.to_csv(f'{data[0].columns[1]}_blend.csv', index_label='id', index=True)

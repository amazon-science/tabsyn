import os

import numpy as np
import pandas as pd

data_path = 'synthetic/sales'

original = pd.read_csv(os.path.join(data_path, 'real.csv'))
synthetic = pd.read_csv(os.path.join(data_path, 'tabsyn.csv'))

total_diff = 0
for column in original.columns:
    print(column)
    if original[column].dtype == 'object':
        diff = np.abs(original[column].value_counts() - synthetic[column].value_counts())
        print(diff.values)
        total_diff += diff.sum()
    else:
        diff = np.abs(original[column].mean() - synthetic[column].mean())
        print(diff)
        total_diff += diff
print('Total')
print(total_diff)
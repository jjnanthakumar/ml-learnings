import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
# print(presidents_df.shape)
# print(presidents_df.info())
# print(len(presidents_df))
# print(type(presidents_df['age']))
# print(type(presidents_df['age'].values))

import numpy as np
x = np.arange(2,8,2)
print(x)
x = np.append(x, x.size)
print(x)
x = np.sort(x)
print(x)


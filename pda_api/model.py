import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('kNN.csv')
x = df.drop(['Drugs Tested', 'Legal Drugs', 'Illegal Drugs', 'ID'], axis=1)
y = df['Illegal Drugs']
kNN = KNeighborsRegressor(n_neighbors=50, weights='distance')
kNN.fit(x, y)

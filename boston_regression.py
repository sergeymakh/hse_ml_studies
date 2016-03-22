import sklearn.datasets
boston_data = sklearn.datasets.load_boston()

X = boston_data['data']
Y = boston_data['target']
strings = len(X)

from sklearn.preprocessing import scale
X = scale(X)

import numpy as np
p_values = np.linspace(1, 10, 200)

from sklearn.cross_validation import KFold
kf = KFold(strings, n_folds = 5, shuffle = True,  random_state=42)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_score

max_acc = -100000
p_max_acc = 0

for p_val in p_values:

    neigh = KNeighborsRegressor(n_neighbors = 5, weights = 'distance', metric = 'minkowski', p = p_val)
    scores = cross_val_score(neigh, X, Y, cv = kf, scoring = 'mean_squared_error')
    #scores = scores*-1
    accuracy =  scores.mean()
    #print accuracy, p_val
    if (accuracy > max_acc):
        max_acc = accuracy
        p_max_acc = p_val

print round(max_acc, 2), round(p_max_acc, 2)
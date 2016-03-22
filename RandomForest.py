import pandas as pd
abalone_data = pd.read_csv("abalone.csv")

abalone_data['Sex'] = abalone_data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = abalone_data[[0, 1, 2, 3, 4, 5, 6, 7]]
Y = abalone_data['Rings']
n_of_elements = len(Y)
print Y
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

min_found = 0.

for n_of_trees in range(1, 50):
    clf = RandomForestRegressor(random_state=1, n_estimators=n_of_trees)
    kf = KFold(n_of_elements, n_folds=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, X, Y, scoring='r2', cv=kf)
    if scores.mean() >= 0.52 and min_found == 0.:
        print "!!!!", n_of_trees, scores.mean()
        min_found = 1
    print n_of_trees, scores.mean()
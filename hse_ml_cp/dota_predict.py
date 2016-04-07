import pandas
import numpy as np
features = pandas.read_csv('./data/data/features.csv', index_col='match_id')
features_test = pandas.read_csv('./data/data/features_test.csv', index_col='match_id')

# prepare for cross-validation
len_features = len(features)
from sklearn.cross_validation import KFold
kf = KFold(n = len_features, n_folds=5, shuffle=True)

#get empty features names
empty_features = features.count()[features.count() < len(features)]
print empty_features

#fill empty fetures with zeros
features = features.fillna(0)
features_test = features_test.fillna(0)
all_cols = features.columns.values.tolist()

#remove result cols
all_cols.remove('duration')
all_cols.remove('radiant_win')
all_cols.remove('tower_status_radiant')
all_cols.remove('tower_status_dire')
all_cols.remove('barracks_status_radiant')
all_cols.remove('barracks_status_dire')


x_cols = all_cols

X_train = features[x_cols]
Y_train = features[['radiant_win']]
X_test = features_test[x_cols]
Y_train = np.ravel(Y_train)

#################################
#### Step 1. Gradient boosting
#################################
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

import time
import datetime


# check score for gradient boosting
"""
print "Gradient boosting cross validation"
# 5, 10, 15, 20, 25, 30, 35, 40,
for n_of_trees in (50, 70, 100):
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators = n_of_trees)
    scores = cross_val_score(clf, X=X_train, y=Y_train, scoring='roc_auc', cv=kf)
    print "n_of_trees:", n_of_trees, ", roc_auc:" , scores.mean(), ', Time elapsed:', datetime.datetime.now() - start_time
"""

# get prediction with gradient boosting
"""
n_of_trees = 30
clf = GradientBoostingClassifier(n_estimators = n_of_trees)
clf.fit(X_train, Y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
print len(y_pred)
y_pred_df = pandas.DataFrame(y_pred, columns=['radiant_win'], index = X_test.index)


print y_pred_df.head()
y_pred_df.to_csv("submit.csv")

"""
#################################
#### Step 2. Logistic regression
#################################
#remove hero cols
"""
all_cols.remove('r1_hero')
all_cols.remove('r2_hero')
all_cols.remove('r3_hero')
all_cols.remove('r4_hero')
all_cols.remove('r5_hero')
all_cols.remove('d1_hero')
all_cols.remove('d2_hero')
all_cols.remove('d3_hero')
all_cols.remove('d4_hero')
all_cols.remove('d5_hero')

x_cols = all_cols

X_train = features[x_cols]
X_test = features_test[x_cols]
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##############
#count unique heroes
hero_ids = np.array([])
for h in xrange(5):
    hero_ids = np.append(hero_ids, np.ravel(features['r%d_hero' % (h+1)]))
    hero_ids = np.append(hero_ids, np.ravel(features['d%d_hero' % (h+1)]))
    hero_ids = np.append(hero_ids, np.ravel(features_test['r%d_hero' % (h+1)]))
    hero_ids = np.append(hero_ids, np.ravel(features_test['d%d_hero' % (h+1)]))

hero_unique = set(hero_ids)
print len(hero_unique)
"""
#add features for heroes in train
X_pick = np.zeros((features.shape[0], 112))
for i, match_id in enumerate(features.index):
    for p in xrange(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
#print len(X_train_scaled), len(X_pick)
X_train_scaled = np.hstack((X_train_scaled, X_pick))

#add features for heroes in test
X_pick_test = np.zeros((features_test.shape[0], 112))
for i, match_id in enumerate(features_test.index):
    for p in xrange(5):
        X_pick_test[i, features_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
#print len(X_test_scaled), len(X_pick_test)
X_test_scaled = np.hstack((X_test_scaled, X_pick_test))
"""
#check score for logistic regression
"""
for c_val in range(1, 50):
    start_time = datetime.datetime.now()
    clf = LogisticRegression(C = c_val)
    scores = cross_val_score(clf, X=X_train_scaled, y=Y_train, scoring='roc_auc', cv=kf)
    #print scores
    print "C:", c_val, ", roc_auc:" , scores.mean(), ', Time elapsed:', datetime.datetime.now() - start_time

"""

# get prediction with logistic regression
"""
c_val = 10
clf = LogisticRegression(C = c_val)
clf.fit(X_train_scaled, Y_train)
y_pred = clf.predict_proba(X_test_scaled)[:, 1]
print len(y_pred)
y_pred_df = pandas.DataFrame(y_pred, columns=['radiant_win'], index = X_test.index)


print y_pred_df.head()
print clf.coef_.size
y_pred_df.to_csv("submit.csv")
"""

for n_of_trees in (50, 70, 100):
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(n_estimators = n_of_trees)
    scores = cross_val_score(clf, X=X_train_scaled, y=Y_train, scoring='roc_auc', cv=kf)
    print "n_of_trees:", n_of_trees, ", roc_auc:" , scores.mean(), ', Time elapsed:', datetime.datetime.now() - start_time
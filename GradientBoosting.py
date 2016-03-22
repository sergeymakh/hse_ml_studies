import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
#%matplotlib inline

gbm_data = pandas.read_csv('gbm-data.csv')
X = gbm_data.values[0:,1:]
y = gbm_data.values[0:,0]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=3600, random_state=241)
clf.fit(X_train, y_train)
y_test_pred = clf.predict_proba(X_test)
test_loss = log_loss(y_test, y_test_pred)
print round(test_loss, 2)
"""
l_rate = 0.2
clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=l_rate)
clf.fit(X_train, y_train)

test_loss = np.zeros(250)
train_loss = np.zeros(250)
min_test_loss = 100
min_test_i = -1
for i, decision in enumerate(clf.staged_decision_function(X_test)):
    y_test_pred = 1.0 / (1.0 + np.exp(-1.*decision))
    test_loss[i] = log_loss(y_test, y_test_pred)
    if (test_loss[i] < min_test_loss):
        min_test_loss = test_loss[i]
        min_test_i = i


for i, decision in enumerate(clf.staged_decision_function(X_train)):
    y_train_pred = 1.0 / (1.0 + np.exp(-1.*decision))
    train_loss[i] = log_loss(y_train, y_train_pred)
"""

"""
plt.figure()

plt.plot(test_loss)
plt.plot(train_loss)
plt.legend(['test score', 'train score'])
plt.show()
#print l_rate
"""
###print min_test_i, round(min_test_loss, 2)
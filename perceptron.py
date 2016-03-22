import pandas

train_data = pandas.read_csv('perceptron-train.csv', header = None)
test_data = pandas.read_csv('perceptron-test.csv', header = None)
X_train = train_data[[1, 2]]
Y_train = train_data[0]
X_test = test_data[[1, 2]]
Y_test = test_data[0]


from sklearn.linear_model import Perceptron
clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc_score_1 = accuracy_score(Y_test, predictions)
acc_score_1

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_train = X_train_scaled
X_test = X_test_scaled
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
acc_score_2 = accuracy_score(Y_test, predictions)
acc_score_2

print acc_score_1, acc_score_2, acc_score_2 - acc_score_1
import pandas

data = pandas.read_csv('svm-data.csv', header = None)
X = data[[1, 2]]
Y = data[0]

from sklearn.svm import SVC
clf = SVC(kernel='linear', C = 100000, random_state=241)
clf.fit(X, Y)
print clf.support_
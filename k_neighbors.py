import pandas

data = pandas.read_csv('wine.data', header = None)
strings =  len(data.index)

X = data[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
Y = data[0]

from sklearn.preprocessing import scale
#X_arr = X.to_array()
#print X_arr
X_scaled = scale(X)
X = pandas.DataFrame(X_scaled)

from sklearn.cross_validation import KFold
kf = KFold(strings, n_folds = 5, shuffle = True,  random_state=42)

max_acc = 0
k_max_acc = 0
from sklearn.neighbors import KNeighborsClassifier
#for nn in range(1, 50):
#    neigh = KNeighborsClassifier(n_neighbors=nn)
#    neigh.fit(X, Y)
#    accuracy = neigh.score(X = X, y = Y)
#    if (accuracy > max_acc):
#        max_acc = accuracy
#        k_max_acc = nn
#print max_acc, k_max_acc

#nn = 3
import numpy
from sklearn.cross_validation import cross_val_score

for nn in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=nn)
    scores = cross_val_score(neigh, X, Y, cv = kf)
    accuracy =  scores.mean()
    #print accuracy, nn
    if (accuracy > max_acc):
        max_acc = accuracy
        k_max_acc = nn
    #acc_arr = numpy.array([])
    #for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
    #    X_train = X.iloc[train_index]
    #    Y_train = Y.iloc[train_index]
    #    X_test = X.iloc[test_index]
    #    Y_test = Y.iloc[test_index]
    #    neigh = KNeighborsClassifier(n_neighbors=nn)
    #    neigh.fit(X_train, Y_train)

        #accuracy = neigh.score(X = X_test, y = Y_test)
        #acc_arr = numpy.append(acc_arr, numpy.array(accuracy))
    #mean_acc = acc_arr.mean()
    #print mean_acc, nn
    #if (accuracy > max_acc):
    #    max_acc = accuracy
    #    k_max_acc = nn
print round(max_acc, 2), k_max_acc
# 1  2 0.83 3

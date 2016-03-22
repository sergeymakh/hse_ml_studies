import pandas

data = pandas.read_csv('classification.csv')
Y_true = data["true"]
Y_pred = data["pred"]
#print Y_true
print len(data[(data.pred == 0) & (data.true == 1)])
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_true, Y_pred)
print cm
#64 34 59 43
#43 34 59 64
print cm[0][0], cm[0][1], cm[1][0], cm[1][1]

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(Y_true, Y_pred)
from sklearn.metrics import precision_score
Precision = precision_score(Y_true, Y_pred)
from sklearn.metrics import recall_score
Recall = recall_score(Y_true, Y_pred)
from sklearn.metrics import f1_score
f_scr = f1_score(Y_true, Y_pred)

print round(Accuracy, 2), round(Precision, 2), round(Recall, 2), round(f_scr, 2)


data = pandas.read_csv('scores.csv')
y_true = data["true"]
score_logreg = data["score_logreg"]
score_svm = data["score_svm"]
score_knn = data["score_knn"]
score_tree = data["score_tree"]

from sklearn.metrics import roc_auc_score
auc_roc_score_logreg = roc_auc_score(y_true, score_logreg)
auc_roc_score_svm = roc_auc_score(y_true, score_svm)
auc_roc_score_knn = roc_auc_score(y_true, score_knn)
auc_roc_score_tree = roc_auc_score(y_true, score_tree)
#print auc_roc_score_logreg, auc_roc_score_svm, auc_roc_score_knn, auc_roc_score_tree
print "score_logreg"

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, score_tree)
#print len(precision)
#print len(recall)
#print len(thresholds)
#res = precision_recall_curve(y_true, score_logreg)
#print precision
#print recall
#print thresholds
#print res
d = {'precision': precision, 'recall': recall}
df = pandas.DataFrame(d)
#print round(df[df.recall > 0.7]['precision'].max(), 2)
"""
score_logreg = 0.63
score_svm = 0.62
score_knn = 0.61
score_tree = 0.65
"""
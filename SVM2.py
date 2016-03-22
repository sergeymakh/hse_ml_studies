from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(data_home="D:\\DataScienceCourse\\scikit", subset='all', categories=['alt.atheism', 'sci.space'])

target = newsgroups.target
#print len(target)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tf_idf = vectorizer.fit_transform(newsgroups.data)
feature_mapping = vectorizer.get_feature_names()
print len(feature_mapping)
#print tf_idf[1785]

import numpy as np
from sklearn.cross_validation import KFold
import sklearn.grid_search as grid_search
import sklearn.svm as svm
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(target.size, n_folds=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241, C = 1)
clf.fit(tf_idf, newsgroups.target)
#clf = svm.SVC(kernel='linear', random_state=241)
#import scipy.sparse
coef_ = clf.coef_.toarray()[0]
print coef_
values = abs(coef_)
print values
top10 = np.argsort(values)[-10:]
print top10
res_indices = top10
words_array = np.array([])
for res_i in res_indices:
    words_array = np.append(words_array, feature_mapping[res_i])
words_array.sort()
words_str = ""
for word in words_array:
    words_str = words_str + " " + word
print words_str
"""
coefs = clf.coef_
print coefs.data
from math import fabs
#coefs2 = coefs.argsort()
#print coefs2
print type(coefs[0])
print coefs[0,1]
print coefs[0,2]
print coefs[0, 2000]
#coef_array = np.array([])
#for i in np.arange(0, coefs.size):
#    coef_array = np.append(coef_array, fabs(np.array(coefs[0, i])))
    #print np.array(coefs[0, i])
#print len(coef_array)


coef_array = abs(coefs.data)
print coef_array
indices = coef_array.argsort()
print indices
#print indices[len(indices)-10:len(indices)]
res_indices = indices[len(indices)-10:len(indices)]
print res_indices
res_indices = np.argsort(coef_array)[-10:]
print res_indices
#res_indices = indices[0:9]
coef_array.sort()
print coef_array
print coef_array.argsort()
print '---'
words_array = np.array([])
for res_i in res_indices:
    words_array = np.append(words_array, feature_mapping[res_i])
words_array.sort()
words_str = ""
for word in words_array:
    words_str = words_str + " " + word
print words_str
"""
# 9043 12942 19069  9667 16027  4187  3694 13550 17802 18158
# 18158 22745 21850 13426  5088 17030  5093 15097 15606 12871
"""
words_array = np.array([])

print feature_mapping[12871]
print feature_mapping[15606]
print feature_mapping[15097]
print feature_mapping[5093]
print feature_mapping[17030]
print feature_mapping[5088]
print feature_mapping[13426]
print feature_mapping[21850]
print feature_mapping[22745]
print feature_mapping[18158]
print '----'
words_array = np.append(words_array, feature_mapping[12871])
words_array = np.append(words_array,  feature_mapping[15606])
words_array = np.append(words_array,  feature_mapping[15097])
words_array = np.append(words_array,  feature_mapping[5093])
words_array = np.append(words_array,  feature_mapping[17030])
words_array = np.append(words_array,  feature_mapping[5088])
words_array = np.append(words_array,  feature_mapping[13426])
words_array = np.append(words_array,  feature_mapping[21850])
words_array = np.append(words_array,  feature_mapping[22745])
words_array = np.append(words_array,  feature_mapping[18158])
words_array.sort()
for word in words_array:
    print word
"""
#print coefs.sort()
#print np.sort(coefs[0])

"""
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#print grid
gs.fit(tf_idf, newsgroups.target)
#print clf.coef_[0]

for a in gs.grid_scores_:
    print a.mean_validation_score
    print a.parameters

print gs.best_score_
print gs.best_estimator_.C
"""
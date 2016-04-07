import numpy as np
from math import exp
from math import log

def w_inner_sum(prev_w1, prev_w2, k, X, Y, priznak_index):
    res_summ = 0.
    l = len(Y)
    #summ_elem = 0.
    for i in np.arange(0, l):
        x1 = float(X[i:i+1][1])
        x2 = float(X[i:i+1][2])
        y = float(Y[i:i+1])
        q = (-1*y*(prev_w1*x1 + prev_w2*x2))
        qq = exp(q)
        summ_elem = 1 - (1/(1 + exp(-1*y*(prev_w1*x1 + prev_w2*x2))))
        if (priznak_index == 1):
            summ_elem = y*x1*summ_elem
        elif (priznak_index == 2):
            summ_elem = y*x2*summ_elem
        else:
            return -999999
        #print summ_elem
        res_summ = res_summ + summ_elem
        #print res_summ
    res_summ = k*(1./l)*res_summ
    return res_summ

def calc_w1(prev_w1, prev_w2, k, X, Y, C):
    w1 = prev_w1 + w_inner_sum(prev_w1, prev_w2, k, X, Y, 1) - k*C*prev_w1
    return w1

def calc_w2(prev_w1, prev_w2, k, X, Y, C):
    w2 = prev_w2 + w_inner_sum(prev_w1, prev_w2, k, X, Y, 2) - k*C*prev_w2
    return w2

def loggistic_value(X, Y, w1, w2, C):
    res_summ = 0.
    for i in np.arange(0, len(Y)):
        x1 = float(X[i:i+1][1])
        x2 = float(X[i:i+1][2])
        y = float(Y[i:i+1])
        summ_elem = log(1 + exp(-1*y*(w1*x1 + w2*x2)))
        res_summ = res_summ + summ_elem
    w = [w1, w2]
    w_norma = np.linalg.norm(w)
    res = res_summ + 0.5*C*w_norma
    return res

"""
w1 = 100
w2 = 200
w = [w1, w2]
print w

res = np.linalg.norm(w)
print res

from math import exp
#def w1(prev_w1, prev_w2, k, X, Y, C):
"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import pandas
data = pandas.read_csv('data-logistic.csv', header = None)
X = data[[1, 2]]
Y = data[0]
#X_train_scaled = scaler.fit_transform(X)
#X = pandas.DataFrame(X_train_scaled)


#print X
"""
i = 1
print int(Y[i:i+1])
print float(X[i:i+1][1])
#k = Y[i:i+1]*X[i:i+1]
#print k
print X[0:1][2]
print X.head()
print Y[2:3]
print Y.head()
"""

"""
priznak_index = 1
prev_w1 = 0
prev_w2 = 0
for i in np.arange(0, len(Y)):
    #summ_elem = Y[i:i+1]*X[i:i+1][priznak_index]*(1/(1 + exp(-1*Y[i:i+1](prev_w1*X[i:i+1][1] + prev_w2*X[i:i+1][2]))))
    (1/(1 + exp(-1*Y[i:i+1](prev_w1*X[i:i+1][1] + prev_w2*X[i:i+1][2]))))
#print summ_elem
"""
#w2 = calc_w2(1, 20, 0.1, X, Y, 2)
#print w2
#w1 = calc_w1(1, 20, 0.1, X, Y, 2)
#print w1

#res = loggistic_value(X, Y, 1, 2, 2)
#print res
#print Y
import math
prev_w1 = 0
prev_w2 = 0
k = 0.1

for step_i in np.arange(0,10000):
    w1 = calc_w1(prev_w1, prev_w2, k, X, Y, 5)
    w2 = calc_w2(prev_w1, prev_w2, k, X, Y, 5)


    e = math.sqrt(math.pow((w2 - prev_w2), 2) + math.pow((w1 - prev_w1), 2))
    if (e - 1e-5) < 0:
        print "i: ", step_i
        break
    print "e: ", e
    #print 1e-5
    #print "diff:", (e - 1e-5)
    #print w1, w2
    prev_w1 = w1
    prev_w2 = w2
print w1, w2

#0.0155638230284 0.0130164988348
#0.0156314193573 0.0130751404568
#0.0148505375692 0.0124085903592
# no regularization
# w1, w2   0.887243829936 0.435042256508
# 0.927
# with regularization
# 0.0144027830964 0.0118760768291
# 0.936
#w1 = 0.887243829936
#w2 = 0.435042256508
a = np.array([])
for i in np.arange(0, len(Y)):
        x1 = float(X[i:i+1][1])
        x2 = float(X[i:i+1][2])
        a_ = 1 / (1 + exp(-w1*x1 - w2*x2))
        a = np.append(a, a_)
print len(Y)
from sklearn.metrics import roc_auc_score
roc = roc_auc_score(Y, a)
print round(roc, 3)



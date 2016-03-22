import pandas as pd
close_prices_data = pd.read_csv("close_prices.csv")
#print close_prices_data.columns.values
col_names = close_prices_data.columns.values[1:len(close_prices_data.columns.values)]
X = close_prices_data[col_names]

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X)
#print pca.components_
#print len(close_prices_data.columns.values)
import numpy as np
rounded_ratio = []
for i in pca.explained_variance_ratio_:
    rounded_ratio = np.append(rounded_ratio, round(i*100, 2))
# 90%  - 4 components
#print (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2])*100
#print rounded_ratio
Y = pca.transform(X)
vec1 = []
for i in np.arange(0, len(Y)):
    vec1 = np.append(vec1, Y[i][0])
#print vec1
#print Y
#print Y.shape
print X.columns[np.argmax(pca.components_[0])]
#print np.argmax(pca.components_[0])
djia_index_data = pd.read_csv("djia_index.csv")
vec2 = djia_index_data['^DJI']
print np.corrcoef(vec1, vec2)
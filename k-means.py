import numpy as np
from skimage.io import imread
image = imread('parrots.jpg')

from skimage import img_as_float
flt_img = img_as_float(image)
"""
X = np.array([])

for i in range(0, flt_img.shape[0]):
    for k in range(0, flt_img.shape[1]):
        X = np.append(X, flt_img[i][k])
    print i, k
print X
"""

img_matrix = np.reshape(flt_img, (flt_img.shape[2], -1))
print img_matrix.shape

from sklearn.cluster import KMeans
clf = KMeans(init='k-means++', random_state=241)
clf.fit(img_matrix)

#import pylab
#pylab.imshow(image)
#pylab.show()
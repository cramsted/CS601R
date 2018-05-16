from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import image_operations as img_op
import numpy as np
from multiprocessing import Pool
import os
from sklearn.svm import LinearSVC
import random

ds = DataSet()
data = ds.butterfly_train

lbp_features = np.empty((0))


def vq(kps, lbp):
    vector = []
    for kp in kps:
        pt = np.around(np.asarray(kp.pt))
        vector.append(int(lbp[int(pt[1]-1), int(pt[0]-1)]))
    return vector


def get_feature(i):
    img = ds.get_image(data[i][1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = img_op.LBP(gray)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return vq(random.sample(kp, int(len(kp)/12)), lbp)


pool = Pool(os.cpu_count())
features = pool.map(get_feature, range(
    int(len(data))))
for feature in features:
    lbp_features = np.hstack((lbp_features, feature))
print(lbp_features.shape)

plt.hist(lbp_features, bins=256)
plt.show()

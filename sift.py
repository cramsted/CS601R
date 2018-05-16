from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import image_operations as img_op
import numpy as np
from multiprocessing import Pool
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC

ds = DataSet()
data = ds.butterfly_train

sift_features = np.empty((0, 128))


def get_feature(i):
    img = ds.get_image(data[i][1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des[np.random.choice(des.shape[0], int(des.shape[0]/12))]


pool = Pool(os.cpu_count())
features = pool.map(get_feature, range(
    int(len(data))))
for feature in features:
    sift_features = np.vstack((sift_features, feature))
print(sift_features.shape)

vq = img_op.vector_quantization(sift_features)
x = plt.hist(vq, bins=200)
clf = LinearSVC()
import pdb
pdb.set_trace()

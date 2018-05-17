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
data_train = ds.butterfly_train
data_test = ds.butterfly_test

lbp_features = np.empty((0))


def LBP(img):
    masks = []
    offsets = [[1, 1],
               [1, 0],
               [1, -1],
               [0, -1],
               [-1, -1],
               [-1, 0],
               [-1, 1],
               [0, 1]]
    ymax, xmax = img.shape
    for x, y in offsets:
        masks.append(img[1+y:ymax-1+y, 1+x:xmax-1+x] > img[1:-1, 1:-1])
    lbp = np.zeros(masks[-1].shape)
    for i, mask in enumerate(masks):
        lbp += mask * 2**i
    return lbp


def make_image_histogram(kps, lbp):
    kpx = np.asarray([np.around(kp.pt[0]) for kp in kps], dtype=np.int32)
    kpy = np.asarray([np.around(kp.pt[1]) for kp in kps], dtype=np.int32)
    vector = lbp[kpy, kpx]
    vals, bins, _ = plt.hist(vector, bins=256, histtype='step')
    return vals


def get_features(args):
    i = args[0]
    data = args[1]
    print(i)
    img = ds.get_image(data[i][1])
    label = data[i][0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = LBP(gray)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return [make_image_histogram(kp, lbp), label]db


def clf_format_data(features):
    X = []
    y = []
    for f, label in features:
        X.append(f)
        y.append(label)
    return X, y


print("Training")
# get_features([0, data_train])
pool = Pool(os.cpu_count())
features = pool.map(get_features, [(i, data_train)
                                   for i in range(len(data_train))])
X, y = clf_format_data(features)
clf = LinearSVC()
clf.fit(X, y)

print("Testing")
pool = Pool(os.cpu_count())
features = pool.map(get_features, [(i, data_test)
                                   for i in range(len(data_test))])
X, y = clf_format_data(features)
predictions = clf.predict(X)
accuracy = np.count_nonzero(np.where(predictions == y)[
                            0]) / predictions.shape[0]
print(accuracy)

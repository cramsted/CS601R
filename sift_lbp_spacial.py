from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import image_operations as img_op
import numpy as np
from multiprocessing import Pool
import os
from sklearn.svm import LinearSVC
import random
from sklearn.metrics import confusion_matrix

ds = DataSet()
data_train = ds.butterfly_train
data_test = ds.butterfly_test

lbp_features = np.empty((0))


def spacial(lbp, kpy, kpx):
    xbound = int(lbp.shape[1]/2)
    ybound = int(lbp.shape[0]/2)
    quadrants = [[0, xbound, 0, ybound],
                 [xbound, lbp.shape[1]+1, 0, ybound],
                 [0, xbound, ybound, lbp.shape[0]+1],
                 [xbound, lbp.shape[1]+1, ybound, lbp.shape[0]+1]]
    vector = np.empty((0))
    for xlower, xupper, ylower, yupper in quadrants:
        mask = np.logical_and(np.logical_and(np.logical_and(
            xlower <= kpx, ylower <= kpy), kpx <= xupper), kpy <= yupper)
        vals, bins, _ = plt.hist(
            lbp[kpy[mask], kpx[mask]], bins=256, histtype='step')
        vector = np.hstack((vector, vals))
    return vector


def make_image_histogram(kps, lbp):
    kpx = np.asarray([np.around(kp.pt[0]) for kp in kps], dtype=np.int32)
    kpy = np.asarray([np.around(kp.pt[1]) for kp in kps], dtype=np.int32)
    return spacial(lbp, kpy, kpx)


def get_features(args):
    i = args[0]
    data = args[1]
    img = ds.get_image(data[i][1])
    label = data[i][0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = img_op.LBP(gray)
    sift = cv2.xfeatures2d.SIFT_create(500)
    kp, des = sift.detectAndCompute(gray, None)
    return [make_image_histogram(kp, lbp), label]


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
print("Accuracy: ", accuracy)
cm = confusion_matrix(y, predictions)
plt.figure()
img_op.plot_confusion_matrix(
    cm, classes=ds.categories, title='SIFT LBP & Spacial Pyramid pooling Confusion Matrix  \nAccuracy={}'.format(accuracy))
plt.show()

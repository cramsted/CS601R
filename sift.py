from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from multiprocessing import Pool
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import pickle

ds = DataSet()
data_train = ds.butterfly_train
data_test = ds.butterfly_test

sift_features = np.empty((0, 128))


def vector_quantization(features):
    mbk = MiniBatchKMeans(n_clusters=200)
    vq = mbk.fit_predict(features)
    vals, bounds, _ = plt.hist(vq, bins=200, histtype='step')
    return vals


def get_feature(args):
    i = args[0]
    data = args[1]
    print(i)
    img = ds.get_image(data[i][1])
    label = data[i][0]
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # for images that don't have enough features
    while des.shape[0] <= 500:
        des = np.vstack((des, des))
    vq = vector_quantization(
        des[np.random.choice(des.shape[0], 500)])
    return [vq, label]


def sort_features(features):
    X = []
    y = []
    feature_count = 0
    for feature, label in features:
        X.append(feature)
        y.append(label)
        feature_count += feature.shape[0]
        import pdb
        pdb.set_trace()
    return X, y, feature_count


pool = Pool(os.cpu_count())
try:
    pickle_rw = open("clf_sift.pickle", "rb")
    clf = pickle.load(pickle_rw)
except:
    # get_feature(324)
    print("Training")
    features = pool.map(get_feature, [(i, data_train)
                                      for i in range(len(data_train))])

    X, y, count = sort_features(features)
    print("Number of features: ", count)
    clf = LinearSVC()
    clf.fit(X, y)
    # save pickle the model
    pickle_rw = open("clf_sift_"+str(count)+".pickle", "wb")
    pickle.dump(clf, pickle_rw)
    pickle_rw.close()

print("Testing")
features = pool.map(get_feature, [(i, data_test)
                                  for i in range(len(data_test))])
X, y, count = sort_features(features)
print("Number of features: ", count)
predictions = clf.predict(X)
accuracy = np.count_nonzero(np.where(predictions == y)[
                            0]) / predictions.shape[0]
print(accuracy)

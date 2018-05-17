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

mbk = MiniBatchKMeans(n_clusters=200)


def vector_quantization_train(features):
    sift_features = np.empty((0, 128))
    for f, label in features:
        mbk.partial_fit(f)
        # sift_features = np.vstack((sift_features, f))
    # mbk.fit(sift_features)
    return vector_quantization(features)


def vector_quantization(features):
    X = []
    y = []
    count = 0
    for f, label in features:
        count += len(f)
        vq = mbk.predict(f)
        vals, bins, _ = plt.hist(vq, bins=200, histtype='step')
        X.append(vals)
        y.append(label)
    return X, y, count


def get_features(args):
    i = args[0]
    data = args[1]
    print(i)
    img = ds.get_image(data[i][1])
    label = data[i][0]
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # return [des[np.random.choice(des.shape[0], 1500)], label]
    return [des, label]


pool = Pool(os.cpu_count())
try:
    pickle_rw = open("clf_sift.pickle", "rb")
    models = pickle.load(pickle_rw)
    clf = models[0]
    mbk = models[1]
except:
    # get_features(324)
    print("Training")
    features = pool.map(get_features, [(i, data_train)
                                       for i in range(len(data_train))])

    X, y, count = vector_quantization_train(features)
    print("Number of features: ", count)
    clf = LinearSVC()
    clf.fit(X, y)
    # save pickle the model
    pickle_rw = open("clf_sift.pickle", "wb")
    pickle.dump([clf, mbk], pickle_rw)
    pickle_rw.close()

print("Testing")
features = pool.map(get_features, [(i, data_test)
                                   for i in range(len(data_test))])
X, y, count = vector_quantization(features)
print("Number of features: ", count)
predictions = clf.predict(X)
accuracy = np.count_nonzero(np.where(predictions == y)[
                            0]) / predictions.shape[0]
print(accuracy)

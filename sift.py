from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from multiprocessing import Pool
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import image_operations as img_op

ds = DataSet()
data_train = ds.butterfly_train
data_test = ds.butterfly_test

mbk = MiniBatchKMeans(n_clusters=400)


def vector_quantization_train(features):
    sift_features = np.empty((0, 128))
    for f, label in features:
        mbk.partial_fit(f)
    return vector_quantization(features)


def vector_quantization(features):
    X = []
    y = []
    count = 0
    for f, label in features:
        count += len(f)
        vq = mbk.predict(f)
        vals, bins, _ = plt.hist(vq, bins=400, histtype='step')
        X.append(vals)
        y.append(label)
    # plt.show()
    plt.close()
    return X, y, count


def get_features(args):
    i = args[0]
    data = args[1]
    img = ds.get_image(data[i][1])
    label = data[i][0]
    sift = cv2.xfeatures2d.SIFT_create(1000)
    kp, des = sift.detectAndCompute(img, None)
    return [des, label]


pool = Pool(os.cpu_count())
print("Training")
features = pool.map(get_features, [(i, data_train)
                                   for i in range(len(data_train))])

X, y, count = vector_quantization_train(features)
print("Number of features: ", count)
clf = LinearSVC()
# clf = RandomForestClassifier()
clf.fit(X, y)

print("Testing")
features = pool.map(get_features, [(i, data_test)
                                   for i in range(len(data_test))])
X, y, count = vector_quantization(features)
print("Number of features: ", count)
predictions = clf.predict(X)
accuracy = np.count_nonzero(np.where(predictions == y)[
                            0]) / predictions.shape[0]
print("Accuracy: ", accuracy)
cm = confusion_matrix(y, predictions)
plt.figure()
img_op.plot_confusion_matrix(
    cm, classes=ds.categories, title='SIFT w/ VQ Confusion Matrix  \nAccuracy={}'.format(accuracy))
plt.show()

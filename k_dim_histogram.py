from DataSet import DataSet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from multiprocessing import Pool
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import image_operations as img_op

ds = DataSet()
data = ds.k_histogram

mbk = MiniBatchKMeans(n_clusters=200)


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
        vals, bins, _ = plt.hist(vq, bins=200, histtype='step')
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
features = pool.map(get_features, [(i, data)
                                   for i in range(len(data))])

X, y, count = vector_quantization_train(features)
X = np.asarray(X).T
y = np.asarray(y)
plt.subplot(121)
plt.imshow(X, cmap='jet', interpolation='nearest')
plt.title("Heat Map of All Histograms")

# X_category = np.ones(X.shape)
# y = np.asarray(y)
# for i in range(1, 11):
#     mask = y == i
#     avg = np.mean(X[:, mask], axis=0)
#     X_category[:, mask] = X_category[:, mask] * avg[:, np.newaxis]
for i in range(1, 11):
    mask = y == i
    avg = np.around(np.mean(X[:, mask], axis=1))
    # import pdb
    # pdb.set_trace()
    for j in range(21):
        if i == 1:
            X_cat = avg
        else:
            X_cat = np.vstack((X_cat, avg))
plt.subplot(122)
plt.imshow(X_cat.T, cmap='jet', interpolation='nearest')
plt.title("Heat Map of Averaged Histograms for each Catgory")
plt.show()

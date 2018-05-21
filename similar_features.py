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
data_train = ds.butterfly_train
data_test = ds.butterfly_test

mbk = MiniBatchKMeans(n_clusters=200)


# def vector_quantization_train(features):
#     sift_features = np.empty((0, 128))
#     for f, label in features:
#         mbk.partial_fit(f)
#     return vector_quantization(features)


# def vector_quantization(features):
#     X = []
#     y = []
#     count = 0
#     for f, label in features:
#         count += len(f)
#         vq = mbk.predict(f)
#         vals, bins, _ = plt.hist(vq, bins=200, histtype='step')
#         X.append(vals)
#         y.append(label)
#     # plt.show()
#     plt.close()
#     return X, y, count


def get_features(args):
    i = args[0]
    data = args[1]
    img = ds.get_image(data[i][1])
    label = data[i][0]
    sift = cv2.xfeatures2d.SIFT_create(300)
    kp, des = sift.detectAndCompute(img, None)
    return [kp, des]


# pool = Pool(os.cpu_count())
print("Training")
# features = pool.map(get_features, [(i, data_train)
#                                    for i in range(len(data_train))])
features = []
for i in range(50):
    features.append(get_features([i, data_train]))

sift_features = np.empty((0))
for kp, des in features:
    mbk.partial_fit(des)

similar_patches = []
for i in np.random.randint(99, size=6):
    patches = []
    for j in range(5):
        kps, dess = get_features([i, data_train])
        for kp, des in zip(kps, dess):
            predict = mbk.predict([des])
            if predict == j:
                patches.append(img_op.getPatchFor(
                    kp, cv2.cvtColor(ds.get_image(data_train[i][1]), cv2.COLOR_BGR2GRAY)))
    similar_patches.append(patches)
images = []
for patches in similar_patches:
    if len(patches) >= 6:
        top_row = np.hstack((patches[0], patches[1]))
        top_row = np.hstack((top_row, patches[2]))
        # top_row = np.hstack((top_row, patches[3]))
        bottom_row = np.hstack((patches[3], patches[4]))
        bottom_row = np.hstack((bottom_row, patches[5]))
        # bottom_row = np.hstack((bottom_row, patches[7]))
        images.append(np.vstack((top_row, bottom_row)))
print(len(images))
for i in range(1, 5):
    plt.subplot(4, 1, i)
    plt.imshow(cv2.cvtColor(images[i-1], cv2.COLOR_GRAY2RGB))
plt.show()

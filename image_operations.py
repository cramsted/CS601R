import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import average_precision_score, precision_recall_curve


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


def dense_sampling(img, padding):
    return img[::padding, ::padding]


def getPatchFor(keypoint, image, out_size=(64, 64)):
    c = np.asarray(keypoint.pt)
    th = 2.0*np.pi * keypoint.angle/360.0
    #sz = keypoint.size / np.sqrt(2)
    sz = keypoint.size / 2.0
    ovec = sz*np.asarray([np.cos(th), np.sin(th)])
    ovec_prp = sz*np.asarray([-np.sin(th), np.cos(th)])
    # out_size =
    pt1 = c + ovec - ovec_prp
    pt2 = c + ovec + ovec_prp
    pt3 = c - ovec - ovec_prp

    src_pts = np.float32([pt1, pt2, pt3])
    targ_pts = np.float32([[0., 0.], [out_size[1], 0.], [0., out_size[0]]])

    tform = cv2.getAffineTransform(src_pts, targ_pts)
    frame = cv2.warpAffine(image, tform, out_size, flags=cv2.INTER_CUBIC)

    return frame


def vector_quantization(features):
    mbk = MiniBatchKMeans(n_clusters=200)
    vq = mbk.fit_predict(features)
    vals, bounds, _ = plt.hist(vq, bins=200, histtype='step')
    return vals


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()


def precision_recall(y_test, y_score):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

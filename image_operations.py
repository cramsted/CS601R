import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


def LBP(img):
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

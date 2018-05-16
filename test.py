from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = LinearSVC(random_state=0)
import pdb
pdb.set_trace()

# from DataSet import DataSet
# import image_operations as img_op
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.cluster import MiniBatchKMeans

# ds = DataSet()
# img = ds.get_image(ds.training_set[0][1])

# # hog
# hog = cv2.HOGDescriptor()
# hog_features = hog.compute(img)

# # miniBatchKMeans
# # mbk = MiniBatchKMeans(n_clusters=200)
# # out_mbk = mbk.fit_predict(hog_features[:10000])
# # import pdb
# # pdb.set_trace()

# # sift
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# frame = img_op.getPatchFor(keypoint=kp[0], image=gray)
# import pdb
# pdb.set_trace()
# plt.imshow(frame)
# plt.show()

# # dense
# # dense = img_op.dense_sampling(img, 8)

# # lbp
# # lbp = img_op.LBP(img)
# # plt.hist(lbp.flatten(), bins=256)
# # plt.show()

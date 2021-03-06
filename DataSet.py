import numpy as np
import cv2
import random
import glob
import json
import copy
import sys


class DataSet:
    def __init__(self):
        # Note: use the miniBatchKMeans for vector quantization
        self.bird_image_paths = glob.glob("../birds/*jpg")

        with open("leeds_dataset_info_corrected.json", "r") as read_file:
            self.data = json.load(read_file)
        self.categories = []
        for i in range(1, 11):
            self.categories.append(self.data["categories"][str(i)])

        self.get_training_set()
        self.get_test_set()
        self.get_k_histogram_set()

    def get_k_histogram_set(self):
        self.k_histogram = []
        for i in range(1, 11):
            for j in range(20):
                self.k_histogram.append(
                    [i, self.data["training_images"][str(i)][j]])

    def get_training_set(self):
        self.butterfly_train = []
        for i in range(1, 11):
            imgs = self.data["training_images"][str(i)]
            for j in imgs:
                self.butterfly_train.append((i, j))

        # self.training_set = copy.copy(self.butterfly_train)
        self.training_set = [[0, path] for label, path in self.butterfly_train]
        for i in range(len(self.butterfly_train)):
            self.training_set.append((1, self.bird_image_paths[i]))

    def get_test_set(self):
        self.butterfly_test = []
        for i in range(1, 11):
            imgs = self.data["test_images"][str(i)]
            for j in imgs:
                self.butterfly_test.append((i, j))

        # self.test_set = copy.copy(self.butterfly_test)
        self.test_set = [[0, path] for label, path in self.butterfly_test]
        for i in range(len(self.butterfly_test)):
            self.test_set.append((1, self.bird_image_paths[i]))

    def get_image(self, path):
        path = "../" + path
        try:
            img = cv2.imread(path, 1)  # color image
            if img is None:
                raise ValueError
        except:
            print("File does not exist!")
            sys.exit()
        ratio = 500 / img.shape[0]
        return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)


if __name__ == '__main__':
    ds = DataSet()

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
        self.bird_image_paths = glob.glob("../birds/birds/*jpg")

        with open("leeds_dataset_info_corrected.json", "r") as read_file:
            self.data = json.load(read_file)
        self.categories = self.data["categories"]

        self.get_training_set()
        self.get_test_set()

    def get_training_set(self):
        self.butterfly_train = []
        for i in range(1, 11):
            imgs = self.data["training_images"][str(i)]
            for j in imgs:
                self.butterfly_train.append((i, j))

        self.training_set = copy.copy(self.butterfly_train)
        for i in range(len(self.butterfly_train)):
            self.training_set.append((11, self.bird_image_paths[i]))

    def get_test_set(self):
        self.butterfly_test = []
        for i in range(1, 11):
            imgs = self.data["test_images"][str(i)]
            for j in imgs:
                self.butterfly_test.append((i, j))

        self.test_set = copy.copy(self.butterfly_test)
        for i in range(len(self.butterfly_test)):
            self.test_set.append((11, self.bird_image_paths[i]))

    def get_image(self, path):
        path = "../leedsbutterfly/" + path
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

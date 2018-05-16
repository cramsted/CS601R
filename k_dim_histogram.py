from DataSet import DataSet

ds = DataSet()

images = []
for i in range(1, 11):
    j = 0
    for img in ds.butterfly_train:
        if img[0] == i:
            images.append(img)
            j += 1
        if j == 10:
            break
import pdb
pdb.set_trace()

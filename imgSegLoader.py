from torch.utils import data
from PIL import Image
import os
import cv2
import torch
import numpy as np

# BACKGROUND = 0
# SKIN = 1
# NOSE = 2
# RIGHT_EYE = 3
# LEFT_EYE = 4
# RIGHT_BROW = 5
# LEFT_BROW = 6
# RIGHT_EAR = 7
# LEFT_EAR = 8
# MOUTH_INTERIOR = 9
# TOP_LIP = 10
# BOTTOM_LIP = 11
# NECK = 12
# HAIR = 13
# BEARD = 14
# CLOTHING = 15
# GLASSES = 16
# HEADWEAR = 17
# FACEWEAR = 18
# IGNORE = 255

class imgSegDataset(data.Dataset):

    def __init__(self):
        super(imgSegDataset, self).__init__()

    def initialize(self, data_path, data_size, is_train):

        self.data_path = data_path
        self.data_size = data_size
        self.train_corpus = []

        for i in range(100):
            index = "%06d" % i
            img = os.path.join(data_path, "{}.png".format(index))
            seg_img = os.path.join(data_path, "{}_seg.png".format(index))

            if os.path.exists(img) and (not is_train or os.path.exists(seg_img)):
                self.train_corpus.append(index)

    def __getitem__(self, index):
        img = os.path.join(self.data_path, "{}.png".format(self.train_corpus[index]))
        seg_img = os.path.join(self.data_path, "{}_seg.png".format(self.train_corpus[index]))

        x = cv2.imread(img)
        y_3 = cv2.imread(seg_img)
        y = y_3[:, :, 0]
        y[y == 255] = 0

        y_cls = []

        for j in range(19):
            if len(np.where(y == j)[0]) > 0:
                y_cls.append(1)
            else:
                y_cls.append(0)
        return (torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float), torch.tensor(y, dtype=torch.long), torch.tensor(y_cls, dtype=torch.float))

    def __len__(self):
        return len(self.train_corpus)
    
    def test_data(self, index):
        img = os.path.join(self.data_path, "{}.png".format(self.train_corpus[index]))

        x = cv2.imread(img)
        return torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float).unsqueeze(0), self.train_corpus[index]

def data_loader(data_path, data_size, batch_size, is_train):
    dataset = imgSegDataset()
    dataset.initialize(data_path, data_size, is_train)

    print("dataset [%s] of size %d was created" %
            (type(dataset).__name__, len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader

from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
from project.Util.DataLoaderUtil import getPathDf
import random

class HgNetDataset(Dataset):
    def __init__(self, conf, input_shape, x_path, y_path):
        self.conf = conf
        # self.classes = classes
        self.input_shape = input_shape
        self._getDataDf(x_path, y_path)
        self.samples = len(self.data_list)
        self.class2idx = {'S':0, 'A':1, 'B':2, 'C':3}

    def _getDataDf(self, x_path, y_path):
        x_pd = getPathDf(path=x_path,
                         ext=self.conf.ext.x)
        y_pd = getPathDf(path=y_path,
                         ext=self.conf.ext.y)

        self.data_list = pd.merge(x_pd, y_pd, how='inner', on='filename')[['filename', 'path_x', 'path_y']].values.tolist()

        random.shuffle(self.data_list)
        random.shuffle(self.data_list)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        _, _x_path, _y_path = self.data_list[idx]
        # _classes = self.conf.output.classes

        _x_img = cv2.imread(_x_path, 1) / 255
        _x_img = cv2.resize(_x_img, (self.input_shape.width, self.input_shape.height))
        _x_img = np.transpose(_x_img, (2, 0, 1))

        _y_label = _y_path.split('\\')[-2]

        _y_label = self.class2idx[_y_label]

        return _x_img, _y_label

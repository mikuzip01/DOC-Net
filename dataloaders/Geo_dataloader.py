import os
import csv
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as tf


class GeoDataset(Dataset):
    def __init__(self, data_csvpath:str,random_flip:bool=True, random_crop:bool=True, crop_box:int=512, transform=None):
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_box = crop_box
        self.csv_path = data_csvpath
        self.dataset = []  # [haze_img_path, clear_img_path]
        self._init_dataset()
        self.transform = transform
        self.__init_transform()

    def _init_dataset(self):
        csv_file = open(self.csv_path, "r")
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            self.dataset.append([row[0], row[1]])
        csv_file.close()

    def __init_transform(self):
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ])

    # def _random_rotate(self, haze, clear):
    #     # 拿到角度的随机数。angle是一个-180到180之间的一个数
    #     angle = transforms.RandomRotation.get_params([-180, 180])
    #     # 对haze和clear图像做相同的旋转操作，保证他们都旋转angle角度
    #     haze = haze.rotate(angle, expand=True)
    #     clear = clear.rotate(angle, expand=True)
    #     return haze, clear

    def _random_flip(self, haze, clear):
        # 50%的概率应用垂直，水平翻转。
        if random.random() > 0.5:
            haze = tf.hflip(haze)
            clear = tf.hflip(clear)
        if random.random() > 0.5:
            haze = tf.vflip(haze)
            clear = tf.vflip(clear)
        return haze, clear

    def _random_crop(self, haze, clear):
        # 50%的概率应用垂直，水平翻转。
        i,j,h,w = transforms.RandomCrop.get_params(haze, (self.crop_box, self.crop_box))
        haze = tf.crop(haze, i,j,h,w)
        clear = tf.crop(clear, i,j,h,w)
        return haze, clear

    def __getitem__(self, item):
        haze = Image.open(self.dataset[item][0]).convert('RGB')
        clear = Image.open(self.dataset[item][1]).convert('RGB')
        if self.random_flip:
            haze, clear = self._random_flip(haze, clear)
        if self.random_crop:
            haze, clear = self._random_crop(haze, clear)

        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze, clear

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    gd = GeoDataset("D:\Dataset\Geographic image\data_train.csv", random_crop=True, random_flip=True)
    gd = GeoDataset("D:\Dataset\Geographic image\data_train.csv", random_crop=False, random_flip=False)
    haze, clear = gd[2]
    utils.save_image((clear + 1) / 2.0, 'clear.jpg')
    utils.save_image((haze + 1) / 2.0, 'haze.jpg')
    pass
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import os
import random
from torchvision import transforms

class brain_train_paired(data.Dataset):
    def __init__(self,t1_dir,t2_dir,split = "train"):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.size = 256
        #self.data_dir = data_dir
        #self.file_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.t1_paths = sorted([os.path.join(self.t1_dir, file) for file in os.listdir(self.t1_dir)])
        self.t2_paths = sorted([os.path.join(self.t2_dir, file) for file in os.listdir(self.t2_dir)])
        self.transform = transforms.Compose([transforms.Resize(self.size)])
    def __len__(self):
        assert len(self.t1_paths) == len(self.t1_paths)
        return len(self.t1_paths)

    def __getitem__(self, idx):
        t1_path = self.t1_paths[idx]
        t2_path = self.t2_paths[idx]
        t1= cv2.imread(t1_path, cv2.IMREAD_UNCHANGED)
        t2 = cv2.imread(t2_path, cv2.IMREAD_UNCHANGED)
        # t1 = cv2.resize(t1, self.size)
        # t2 = cv2.resize(t2, self.size)
        t1 = torch.tensor(t1).unsqueeze(0).float()/255.
        t2 = torch.tensor(t2).unsqueeze(0).float()/255. 
        t1 = self.transform(t1)
        t2 = self.transform(t2)

        return {'t1':t1, 't2':t2}
    

class brain_train_single(data.Dataset):
    def __init__(self,img_dir,split = "train"):
        self.img_dir = img_dir
        #self.data_dir = data_dir
        #self.file_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.img_paths = sorted([os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)])
        #print('ok')
    def __len__(self):
        #assert 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img= cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        img = torch.tensor(img).unsqueeze(0).float()/255.

        return {'img':img }
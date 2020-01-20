from __future__ import print_function, division

import time
from ast import literal_eval

import os
import random
import torch
import pandas as pd
import numpy as np
import yaml
import h5py
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

class MPR_Dataset(Dataset):

    def __init__(self, patient_dict, transform=None):
        self.arteries, self.dicom_paths = zip(*patient_dict.items())
        print(self.arteries)
        self.transform = transform

    def __len__(self):
        return len(self.arteries)

    def __getitem__(self, idx):
        dicom_paths = self.dicom_paths[idx]
        artery_name = self.arteries[idx]
        images = []
        for dicom_path in dicom_paths:
            dcm_file = pydicom.dcmread(dicom_path)
            cur_img = cv2.normalize(dcm_file.pixel_array, None, alpha = 0, 
                beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            if self.transform:
                cur_img = self.transform(cur_img)
                cur_img = cur_img.repeat(3, 1, 1).type(torch.FloatTensor)
            images.append(cur_img)

        images = torch.stack(images)
        return images, artery_name
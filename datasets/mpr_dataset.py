from __future__ import print_function, division

import time
from ast import literal_eval

import os
import random
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import yaml
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2


class MPR_Dataset(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)
        return X, y


class MPR_Dataset_New_Test(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def remove_borders(self, img):
        vertical_borders = np.where(~((img == 0).sum(0) > img.shape[0] * 0.8))[0]
        horizontal_borders = np.where(~((img == 0).sum(1) > img.shape[1] * 0.8))[0]
        
        without_horizontal_borders = img[horizontal_borders, :]
        without_any_borders = without_horizontal_borders[:, vertical_borders]
        return without_any_borders

    def remove_text(self, img):
        mask = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)[1]#[:,:,0]
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        dst = cv2.inpaint(img, dilated_mask, 5, cv2.INPAINT_NS)
        return dst

    def preprocess_image(self, img):
        removed_borders = self.remove_borders(img)
        removed_text = self.remove_text(removed_borders)
        final_result = cv2.resize(removed_text,(512,512), interpolation = cv2.INTER_LINEAR)
        return final_result

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path,0)

        X = self.preprocess_image(X)
        X = cv2.cvtColor(X,cv2.COLOR_GRAY2RGB)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)
        return X, y


class MPR_Dataset_H5(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]

        # read image
        X = self.__read_h5_image(path)
        # remove text
        X[X == X.max()] = X.min()
        # Segment
        # X = self.__primitive_segmentation(X)
        # convert to another range
        X = self.__scale(X, out_range=(0, 1))
        # X = np.interp(X, (X.min(), X.max()), (0, 1))

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)

        # create 3 channels
        X = X.repeat(3, 1, 1).type(torch.FloatTensor)
        return X, y

    def __scale(self, x, out_range=(-1, 1), axis=None):
        domain = np.min(x, axis), np.max(x, axis)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

    def __preprocess_mask(self, mask):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        min_size = 150
        
        new_mask = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                new_mask[output == i + 1] = 1
        return new_mask

    def __primitive_segmentation(self, img):
        mask = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
        mask[(img > 150) & (img < 776)] = 1
        preprocessed_mask = self.__preprocess_mask(mask)
        kernel = np.ones((10,10), np.uint8) 
        preprocessed_mask = cv2.dilate(preprocessed_mask, kernel, iterations=1) 
        modernized_img = np.copy(img)
        modernized_img[preprocessed_mask == 0] = modernized_img.min() 
        return modernized_img

    def __read_h5_image(self, path_to_file):
        with h5py.File(path_to_file, 'r') as hf:
            img = hf['X'][:]
        return img

class MPR_Dataset_STENOSIS_REMOVAL(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'

    def __init__(self, root_dir, partition="train", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        df = df[
                    (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                    (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
               ]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
        self.df = df
        self.__filter_stenos()

    def __filter_stenos(self):
        mapper = \
            {
                'NORMAL': 0, '-': 0, 
                '25%': 1, '<25%': 1,
                '*50%': 2, '>50%': 2, '70%': 2, '50-70%': 2, '50%': 2, '>70%': 2, '90%': 2,  '90-100%': 2, 
                '75%': 2, '>75%': 2, '70-90%': 2, '>90%': 2, '25-50%': 2, '<50%': 2, '<35%': 2,
            }
        remove_elements = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.df = self.df[pd.Series(remove_elements, index=self.df.index) != 1]

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        self.labels = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x])).tolist()
        self.arteries = self.df[self.ARTERY_COLUMN].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, info[self.IMG_PATH_COLUMN])
        artery = self.arteries[idx]
        stenosis_scores = info[self.STENOSIS_SCORE_COLUMN]
        y = self.labels[idx]
        X = cv2.imread(path)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)
        return X, y

class MPR_Dataset_LSTM(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'
    SEGMENT_ID_COLUMN = "SEGMENT_ID"

    def __init__(self, root_dir, partition="train", level="img", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__detect_segments()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    # def __load_data(self):
    #     df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
    #     if 'filters' in self.config:
    #         df = df[
    #                     (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
    #                     (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
    #                ]
    #     df = df[~df['IMG_PATH'].str.contains('CTCALEK24101973/PLV_RCA/')]
    #     df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)
    #     self.df = df

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        if 'filters' in self.config:
            df = df[
                        (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                        (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
                   ]
        df = df[~df['IMG_PATH'].str.contains('CTCALEK24101973/PLV_RCA/')]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)

        # Filter all stenosis score values which are in dataframe, but not included for training
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        temp = df[self.STENOSIS_SCORE_COLUMN].apply(
            lambda x: np.mean([True if el in mapper.keys() else False for el in x]))
        indeces = temp[temp == 1].index
        self.df = df.ix[indeces]

    def __detect_segments(self):
        self.df[self.SEGMENT_ID_COLUMN] = self.df[self.IMG_PATH_COLUMN].str.split("/").\
            apply(lambda x: x[-1].rsplit("_", maxsplit=1)[0]).factorize()[0]

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group

        self.df[self.LABEL_COLUMN] = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x]))
        self.labels = self.df.groupby(by=self.SEGMENT_ID_COLUMN)[self.LABEL_COLUMN].max().tolist()

    def __len__(self):
        # TODO: Add image length
        return len(self.df[self.SEGMENT_ID_COLUMN].unique())

    def __getitem__(self, idx):
        # TODO: Add image mask
        mask = self.df[self.SEGMENT_ID_COLUMN] == idx
        df_masked = self.df[mask]

        img_pathes = df_masked[self.IMG_PATH_COLUMN]
        images = []
        state = random.getstate()
        for img_path in img_pathes:
            img_path = os.path.join(self.root_dir, self.partition, img_path)
            img = cv2.imread(img_path)

            if self.augmentation:
                random.setstate(state)
                img = self.augmentation(img)

            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)

        y = self.labels[idx]
        viewpoint_indexes = torch.tensor(df_masked[self.VIEWPOINT_INDEX_COLUMN][mask].tolist())
        X = images[viewpoint_indexes.argsort()]

        return X, y


class MPR_Dataset_LSTM_H5(Dataset):
    LABELS_FILENAME = "labels.csv"

    ARTERY_COLUMN = "ARTERY_SECTION"
    VIEWPOINT_INDEX_COLUMN = "MPR_VIEWPOINT_INDEX"
    IMG_PATH_COLUMN = 'IMG_PATH'
    STENOSIS_SCORE_COLUMN = 'STENOSIS_SCORE'
    LABEL_COLUMN = 'LABEL'
    SEGMENT_ID_COLUMN = "SEGMENT_ID"

    def __init__(self, root_dir, partition="train", level="img", transform=None, augmentation=None, config={}):
        self.root_dir = root_dir
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__detect_segments()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))
        if 'filters' in self.config:
            df = df[
                        (df[self.ARTERY_COLUMN].isin(self.config['filters']["arteries"])) &
                        (df[self.VIEWPOINT_INDEX_COLUMN] % self.config['filters']["viewpoint_index_step"] == 0)
                   ]
        df = df[~df['IMG_PATH'].str.contains('CTCALEK24101973/PLV_RCA/')]
        df[self.STENOSIS_SCORE_COLUMN] = df[self.STENOSIS_SCORE_COLUMN].apply(literal_eval)

        # Filter all stenosis score values which are in dataframe, but not included for training
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group
        temp = df[self.STENOSIS_SCORE_COLUMN].apply(
            lambda x: np.mean([True if el in mapper.keys() else False for el in x]))
        indeces = temp[temp == 1].index
        self.df = df.ix[indeces]

    def __detect_segments(self):
        self.df[self.SEGMENT_ID_COLUMN] = self.df[self.IMG_PATH_COLUMN].str.split("/").\
            apply(lambda x: x[-1].rsplit("_", maxsplit=1)[0]).factorize()[0]

    def __find_labels(self):
        mapper = {}
        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group

        self.df[self.LABEL_COLUMN] = self.df[self.STENOSIS_SCORE_COLUMN].apply(lambda x: max([mapper[el] for el in x]))
        self.labels = self.df.groupby(by=self.SEGMENT_ID_COLUMN)[self.LABEL_COLUMN].max().tolist()

    def __len__(self):
        # TODO: Add image length
        return len(self.df[self.SEGMENT_ID_COLUMN].unique())

    def __getitem__(self, idx):
        # TODO: Add image mask
        mask = self.df[self.SEGMENT_ID_COLUMN] == idx
        df_masked = self.df[mask]

        img_pathes = df_masked[self.IMG_PATH_COLUMN]
        images = []
        state = random.getstate()
        for img_path in img_pathes:
            img_path = os.path.join(self.root_dir, self.partition, img_path)

            # read image
            img = self.__read_h5_image(img_path)
            # remove text
            img[img == img.max()] = img.min()
            # convert to another range
            img = np.interp(img, (img.min(), img.max()), (0, 1))

            if self.augmentation:
                random.setstate(state)
                img = self.augmentation(img)

            if self.transform:
                img = self.transform(img)

            # create 3 channels
            img = img.repeat(3, 1, 1).type(torch.FloatTensor)

            images.append(img)
        images = torch.stack(images)

        y = self.labels[idx]
        viewpoint_indexes = torch.tensor(df_masked[self.VIEWPOINT_INDEX_COLUMN][mask].tolist())
        X = images[viewpoint_indexes.argsort()]

        return X, y

    def __read_h5_image(self, path_to_file):
        with h5py.File(path_to_file, 'r') as hf:
            img = hf['X'][:]
        return img


if __name__ == '__main__':
    root_dir = '/home/petryshak/CoronaryArteryPlaqueIdentification/data/all_branches_with_pda_plv_h5'

    with open('../config.yaml', 'r') as f:
       config = yaml.load(f, Loader=yaml.FullLoader)

    import inspect
    import importlib

    def __module_mapping(module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping

    def __load_augmentation(config):
        if 'augmentation' in config['data']:
            mapping = __module_mapping('augmentations')
            augmentation = mapping[config['data']['augmentation']['name']](
                **config['data']['augmentation']['parameters'])
        else:
            augmentation = None
        return augmentation

    def __load_sampler(sampler_name):
        mapping = __module_mapping('samplers')
        sampler = mapping[sampler_name]
        return sampler

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    augmentation = __load_augmentation(config)
    sampler= __load_sampler('ImbalancedDatasetSampler') # 
    dataset = MPR_Dataset_H5(root_dir, config=config["data"], augmentation=augmentation, transform=transform)

    # print(np.unique(np.array(dataset.labels), return_counts=True))
    train_loader = DataLoader(dataset, sampler=sampler(dataset), batch_size=6)

    for img, label in train_loader:
        print(label)
"""
Code essentially copy-paste from the pmf repository:  https://github.com/hushell/pmf_cvpr22/blob/4d7502fe0ea4ffdc4ee9c7d7407a3f6c19b1f208/datasets/episodic_dataset.py
"""

import torch
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import pickle
from pathlib import Path


def PilLoaderRGB(imgPath) :
    return Image.open(imgPath).convert('RGB')


class EpisodeDataset(data.Dataset):
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nCls = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nCls: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """
    def __init__(self, imgDir, nCls, nSupport, nQuery, nEpisode=2000):
        super().__init__()

        self.loaded_dict = imgDir
        self.clsList = sorted(list(self.loaded_dict.keys()))
        
        self.nSamples = 0
        for cls in self.loaded_dict:
            self.nSamples += len(self.loaded_dict[cls])
        self.nCls = nCls
        self.nSupport = nSupport
        self.nQuery = self.nSamples - (nCls * nSupport)
        self.nEpisode = nEpisode

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(nCls * nSupport, 384)
        self.labelSupport = intType(nCls * nSupport)
        self.tensorQuery = floatType(self.nQuery, 384)
        self.labelQuery = intType(self.nQuery)
        self.support_img_files = np.empty(nCls * nSupport, dtype='object')
        self.query_img_files = np.empty(self.nQuery, dtype='object')
        self.mapCls = {}

        # labels {0, ..., nCls-1}
        last_query_idx = 0
        for i, cls in enumerate(self.clsList):
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            len_samples = len(self.loaded_dict[cls]) - self.nSupport
            self.labelQuery[last_query_idx : last_query_idx + len_samples] = i
            last_query_idx += len_samples
        
        for i, cls in enumerate(self.clsList):
            self.mapCls[cls] = i
        
    def __len__(self):
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Return an episode

        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery
                       'LabeltoClass': dict mapping class label with class name}
        """
        # select nCls from clsList
        clsEpisode = self.clsList
        LabeltoClass = dict()
        last_query_idx = 0
        for i, cls in enumerate(clsEpisode):
            #dict LabeltoClass
            LabeltoClass[i] = cls
            
            imgList = self.loaded_dict[cls]
            n_samples_query = len(self.loaded_dict[cls]) - self.nSupport

            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(len(imgList), n_samples_query + self.nSupport, replace=False)
            imgCls = [imgList[i] for i in imgCls]
            
            for j in range(self.nSupport) :
                img = imgCls[j]
                self.support_img_files[i * self.nSupport + j] = img[1]
                self.tensorSupport[i * self.nSupport + j] = img[0]

            for j in range(n_samples_query) :
                img = imgCls[j + self.nSupport]
                self.query_img_files[last_query_idx + j] = img[1]
                self.tensorQuery[last_query_idx + j] = img[0]

            last_query_idx += n_samples_query
            
        ## Random permutation. Though this is not necessary in our approach
        permSupport = torch.randperm(self.nCls * self.nSupport)
        permQuery = torch.randperm(self.nQuery)

        img_file_support = list(self.support_img_files[permSupport])
        img_file_query = list(self.query_img_files[permQuery])
        
        return (self.tensorSupport[permSupport],
               self.labelSupport[permSupport],
               self.tensorQuery[permQuery],
               self.labelQuery[permQuery],
               img_file_support,
               img_file_query,
               LabeltoClass)

class InferenceDataset(data.Dataset):
    """

    :param string imgDir: image directory, each category is in a sub file;
    :param transform: image transformation/data augmentation;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """
    def __init__(self, dataset_dict, transform, inputW, inputH):
        super().__init__()
        
        self.loaded_dict = dataset_dict
        self.clsList = sorted(list(self.loaded_dict.keys()))
            
        self.transform = transform

        floatType = torch.FloatTensor

        self.imgTensor = floatType(3, inputW, inputH)
        self.mapCls = {}
        self.labelToClass = {}
        
        self.len = 0
        for i, cls in enumerate(self.clsList):
            self.mapCls[cls] = i
            self.labelToClass[i] = cls
            self.len += len(self.loaded_dict[cls])

        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Return a sample

        :return dict: {'Image': 1 x nSupport x 3 x H x W,
                       'Label': 1 x nSupport,
                       'LabeltoClass': dict mapping class label with class name}
        """
        # select nCls from clsList
        
        cur_idx = idx
        img_file = None

        for cls in self.clsList:
            if len(self.loaded_dict[cls]) > cur_idx: 
                img = self.loaded_dict[cls][cur_idx]
                label = self.mapCls[cls]
                break

            cur_idx -= len(self.loaded_dict[cls])
        
        if isinstance(img, str) or isinstance(img, Path):
            img_file = str(img)
            I = PilLoaderRGB(img_file)
            img = self.transform(I)
        
        if isinstance(img, tuple):
            img_file = img[1]
            img = img[0]

        return (img,
               label,
               img_file,
               self.labelToClass)

class SupportDataset(data.Dataset):
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nCls = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nCls: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param transform: image transformation/data augmentation;
    """
    def __init__(self, dataset_dict, nCls, nSupport, transform, nEpisode=2000):
        super().__init__()

        self.loaded_dict = dataset_dict
        
        self.clsList = sorted(list(self.loaded_dict.keys()))
            
        self.nCls = nCls
        self.nSupport = nSupport
        self.transform = transform
        self.nEpisode = nEpisode

        floatType = torch.FloatTensor
        intType = torch.LongTensor

        self.tensorSupport = floatType(nCls * nSupport, 384)
        self.labelSupport = intType(nCls * nSupport)
        self.img_files = np.empty(nCls * nSupport, dtype='object')
        self.mapCls = {}

        # labels {0, ..., nCls-1}
        for i in range(self.nCls):
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
           
        for i, cls in enumerate(self.clsList):
            self.mapCls[cls] = i
        
    def __len__(self):
        return self.nEpisode

    def __getitem__(self, idx):
        """
        Return an episode

        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery
                       'LabeltoClass': dict mapping class label with class name}
        """
        # select nCls from clsList
        clsEpisode = self.clsList
        LabeltoClass = dict()
        
        for i, cls in enumerate(clsEpisode):
            #dict LabeltoClass
            LabeltoClass[i] = cls
            
            imgList = self.loaded_dict[cls]

            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(len(imgList), self.nSupport, replace=False)
            imgCls = [imgList[i] for i in imgCls]

            for j in range(self.nSupport) :
                img = imgCls[j]
                self.tensorSupport[i * self.nSupport + j] = img[0]
                self.img_files[i * self.nSupport + j] = img[1]
    
        ## Random permutation. Though this is not necessary in our approach
        permSupport = torch.randperm(self.nCls * self.nSupport)

        img_files = list(self.img_files[permSupport])
        
        return (self.tensorSupport[permSupport],
               self.labelSupport[permSupport],
               img_files,
               LabeltoClass)
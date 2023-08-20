import numpy as np
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd

from os import listdir
from pathlib import Path


def dataset_setting(dataset_path, img_size=32):
    """
    Return dataset setting

    :param int nSupport: number of support examples
    """
    mean =  [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
  

    valTransform = transforms.Compose([#lambda x: np.asarray(x),
                                       transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       normalize])
    inputW, inputH = img_size, img_size

    if dataset_path.endswith('pkl'):
         with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)

    elif dataset_path.endswith('csv'):
        dataset_dict = transform_dataset(dataset_path)

    elif os.path.isdir(dataset_path):
        dataset_path = Path(dataset_path)
        classes = sorted(listdir(dataset_path))

        dataset_dict = {}
        for cls in classes:
            dataset_dict[cls] = list(map(lambda x: dataset_path / cls / x, listdir(dataset_path / cls)))

  
    return valTransform, inputW, inputH, dataset_dict

def transform_dataset(csv_path):
    dataset_dict = {}
    df = pd.read_csv(csv_path)
    classes = df['label'].unique()
    for cls in classes:
        images = df.loc[df['label'] == cls]['filename'].tolist()
        dataset_dict[cls]= images

    return dataset_dict
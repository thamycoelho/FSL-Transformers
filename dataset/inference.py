import numpy as np
import torchvision.transforms as transforms
import os
import pickle
import pandas as pd

from pathlib import Path


def dataset_setting(dataset_path, csv_file=None, csv_sep=None, img_size=32, dataset_name=""):
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

    if csv_file:
        dataset_dict = transform_dataset(csv_file, dataset_path, csv_sep, dataset_name=dataset_name)
    
    elif dataset_path.endswith('pkl'):
        with open(dataset_path, 'rb') as f:
            dataset_dict = pickle.load(f)

    elif os.path.isdir(dataset_path):
        dataset_path = Path(dataset_path)
        classes = sorted(os.listdir(dataset_path))
        dataset_dict = {}
        for cls in classes:
            if os.path.isdir((dataset_path / cls)):
                img_files = filter(lambda item: item.is_file(), (dataset_path / cls).rglob("*")) 
                if cls not in dataset_dict:
                    dataset_dict[cls] = []
                dataset_dict[cls].extend(img_files)

    return valTransform, inputW, inputH, dataset_dict

def transform_dataset(csv_file, data_path, sep, dataset_name=None):
    dataset_dict = {}
    df = pd.read_csv(csv_file, header=None, names=['filename', 'label'], sep=sep)
    classes = df['label'].unique()
    support_classes = classes = list(map(lambda x: x.lower(), classes))
    if dataset_name == "csam_indoor":
        support_classes = ['bathroom', 'bedroom', 'childs_room', 'classroom', 'dressing_room', 'living_room', 'studio', 'swimming_pool']
    
    for cls in classes:
        if cls in support_classes:
            images = df.loc[df['label'].str.lower() == cls]['filename'].tolist()
            print(images)
            images = list(map(lambda file: os.path.join(data_path, file), images))
            if not cls in dataset_dict:
                dataset_dict[cls] = []
            dataset_dict[cls].extend(images)

    return dataset_dict
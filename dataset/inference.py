import numpy as np
import torchvision.transforms as transforms

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

    dataset_path = Path(dataset_path)
    classes = sorted(listdir(dataset_path))

    dataset_dict = {}
    for cls in classes:
        dataset_dict[cls] = list(map(lambda x: dataset_path / cls / x, listdir(dataset_path / cls)))

  
    return valTransform, inputW, inputH, dataset_dict

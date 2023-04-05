from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import collections
import scipy 
import os
import random
import seaborn as sns

def _save_confusion_matrix(cm, target_names,
                      save_path,
                      title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    size = cm.shape[0]
    fig, ax = plt.subplots(figsize=(size + 5,size + 2))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_path / "confusion_matrix.png")

def generate_confusion_matrix(y_true, y_pred, label_names, path):
    
    cf_matrix = confusion_matrix(y_true, y_pred)

    _save_confusion_matrix(cf_matrix, label_names, path)
    
def map_labels(class_to_label, label_to_class, labels):
    cls = [label_to_class[x.item()] for x in labels]
    return [class_to_label[x[0]] for x in cls]

def init_seed(seed=0, deterministic=False):
    """

    :param seed:
    :param deterministic:
    :return:
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
def prepare_device(device_ids, n_gpu_use):
    """

    :param n_gpu_use:
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids)

    n_gpu = torch.cuda.device_count()
    print("gpu count", n_gpu)
    if n_gpu_use > 0 and n_gpu == 0:
        print("the model will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(
            "only {} are available on this machine, "
            "but the number of the GPU in config is {}.".format(n_gpu, n_gpu_use)
        )
        n_gpu_use = n_gpu

    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))

    return device, list_ids

def get_device(args):
    """
    Init the devices from the config file.

    Args:
        config (dict): Parsed config file.

    Returns:
        tuple: A tuple of deviceand list_ids.
    """
    init_seed(args.seed, args.deterministic)
    device, list_ids = prepare_device(args.device_ids, args.n_gpu)
    return device, list_ids

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")
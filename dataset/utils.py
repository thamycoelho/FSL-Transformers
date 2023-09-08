import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .episodic_dataset import EpisodeDataset, InferenceDataset, SupportDataset


def get_sets(args):
    dataset = torch.load(args.support_file, map_location='cpu')
    
    testSet = EpisodeDataset(imgDir = dataset,
                            nCls = len(dataset.keys()),
                            nSupport = args.nSupport,
                            nQuery = args.nQuery,
                            nEpisode = args.nEpisode)

    return testSet

def get_inference_set(args):
    from .inference import dataset_setting

    csv_file = args.csv_file if args.csv_file else None
    
    # If not meta_dataset
    transform, inputW, inputH, testDict = \
        dataset_setting(args.dataset_path, csv_file=csv_file, img_size=args.img_size, csv_sep=args.csv_sep, dataset_name=args.dataset)

    testSet = InferenceDataset(dataset_dict=testDict, 
                               transform=transform, 
                               inputW=inputW, 
                               inputH=inputH)

    return testSet

def get_classifier_set(args):
    support_file = args.support_file
    query_file = args.query_file

    support = torch.load(support_file, map_location='cpu')
    query = torch.load(query_file, map_location='cpu')

    testSet = InferenceDataset(dataset_dict=query, 
                               transform=None, 
                               inputW=1, 
                               inputH=1)

    supportSet = SupportDataset(dataset_dict=support,
                                nCls = len(support.keys()),
                                nSupport =  args.nSupport,
                                transform = None,
                                nEpisode = args.nEpisode)

    return supportSet, testSet

def get_loaders(args):
    print("get datasets")
    # datasets
    if args.classify and args.eval_general:
        dataset_train, dataset_vals = get_classifier_set(args)
    elif args.classify and args.eval_fsl:
        dataset_vals = get_sets(args)
    elif args.extract_features:
        dataset_vals = get_inference_set(args)
    
    global_labels_val = dataset_vals.mapCls
    if args.classify and args.eval_general:
        global_labels_val = dataset_train.mapCls
        
    # Val loader
    print("prepare val dataset") 
    
    sampler_val = SequentialSampler(dataset_vals)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_val = DataLoader(
        dataset_vals, 
        sampler=sampler_val if args.seed > -1 else None,
        batch_size=args.batch_size,
        num_workers=1, # more workers can take too much CPU
        pin_memory=args.pin_mem,
        drop_last=False,
        generator=generator if args.seed > -1 else None
    )

    if args.extract_features or (args.classify and args.eval_fsl):
        return None, data_loader_val, global_labels_val

    # Train loader
    print("prepare train dataset")
    sampler_train = RandomSampler(dataset_train)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train if args.seed > -1 else None,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        generator= generator if args.seed > -1 else None
    )

    return data_loader_train, data_loader_val, global_labels_val
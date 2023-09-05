import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .episodic_dataset import EpisodeDataset, InferenceDataset, SupportDataset


def get_sets(args):
    if args.dataset == 'places':
        from .places import dataset_setting
    elif args.dataset == 'places_600':
        from .places600 import dataset_setting
    elif args.dataset == 'test':
        from .test import dataset_setting
    elif args.dataset == "final_test":
        from .places600_final import dataset_setting

    else:
        raise ValueError(f'{args.dataset} is not supported.')

    # If not meta_dataset
    trainTransform, valTransform, inputW, inputH, \
    trainDir, valDir, testDir, _, _ = \
            dataset_setting(args.nSupport, args.img_size)
    
    trainSet = valSet = testSet = None

    if trainDir:
        trainSet = EpisodeDataset(imgDir = trainDir,
                                nCls = args.nClsEpisode,
                                nSupport = args.nSupport,
                                nQuery = args.nQuery,
                                transform = trainTransform,
                                inputW = inputW,
                                inputH = inputH,
                                nEpisode = args.nEpisode)

    if valDir:
        valSet = EpisodeDataset(imgDir = valDir,
                        nCls = args.nClsEpisode,
                        nSupport = args.nSupport,
                        nQuery = args.nQuery,
                        transform = valTransform,
                        inputW = inputW,
                        inputH = inputH,
                        nEpisode = args.nEpisode)

    if testDir:
        testSet = EpisodeDataset(imgDir = testDir,
                                nCls = args.nClsEpisode,
                                nSupport = args.nSupport,
                                nQuery = args.nQuery,
                                transform = valTransform,
                                inputW = inputW,
                                inputH = inputH,
                                nEpisode = args.nEpisode)
    
    return trainSet, valSet, testSet

def get_inference_set(args):
    from .inference import dataset_setting

     # If not meta_dataset
    transform, inputW, inputH, testDict = \
        dataset_setting(args.dataset_path, args.img_size)
    
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
    if args.classify:
        dataset_train, dataset_vals = get_classifier_set(args)
    elif args.extract_features:
        dataset_vals = get_inference_set(args)
    elif args.eval:
        _, _, dataset_vals = get_sets(args)
    else:
        dataset_train, dataset_vals, _ = get_sets(args)
    
    global_labels_val = dataset_vals.mapCls

    if args.classify:
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

    if args.eval or args.extract_features:
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
import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .episodic_dataset import EpisodeDataset


def get_sets(args):
    if args.dataset == 'places':
        from .places import dataset_setting
    elif args.dataset == 'places_600':
        from .places600 import dataset_setting
    elif args.dataset == 'test':
        from .test import dataset_setting
    else:
        raise ValueError(f'{args.dataset} is not supported.')

    # If not meta_dataset
    trainTransform, valTransform, inputW, inputH, \
    trainDir, valDir, testDir, episodeJson, nbCls = \
            dataset_setting(args.nSupport, args.img_size)
    
    trainSet = EpisodeDataset(imgDir = trainDir,
                              nCls = args.nClsEpisode,
                              nSupport = args.nSupport,
                              nQuery = args.nQuery,
                              transform = trainTransform,
                              inputW = inputW,
                              inputH = inputH,
                              nEpisode = args.nEpisode)
    
    valSet = EpisodeDataset(imgDir = valDir,
                     nCls = args.nClsEpisode,
                     nSupport = args.nSupport,
                     nQuery = args.nQuery,
                     transform = valTransform,
                     inputW = inputW,
                     inputH = inputH,
                     nEpisode = args.nEpisode)

    testSet = EpisodeDataset(imgDir = testDir,
                             nCls = args.nClsEpisode,
                             nSupport = args.nSupport,
                             nQuery = args.nQuery,
                             transform = valTransform,
                             inputW = inputW,
                             inputH = inputH,
                             nEpisode = args.nEpisode)
    
    return trainSet, valSet, testSet


def get_loaders(args):
    print("get datasets")
    # datasets
    if args.eval:
        _, _, dataset_vals = get_sets(args)
    else:
        dataset_train, dataset_vals, _ = get_sets(args)
    
    global_labels_val = dataset_vals.mapCls

    # Val loader
    print("prepare val dataset")
    if not isinstance(dataset_vals, dict):
        dataset_vals = {'single': dataset_vals}
        
    data_loader_val = {}
    
    for j, (source, dataset_val) in enumerate(dataset_vals.items()):
        sampler_val = SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        data_loader = DataLoader(
            dataset_val, 
            sampler=sampler_val if args.seed > -1 else None,
            batch_size=1,
            num_workers=3, # more workers can take too much CPU
            pin_memory=args.pin_mem,
            drop_last=False,
            generator=generator if args.seed > -1 else None
        )
        data_loader_val[source] = data_loader

    if 'single' in dataset_vals:
        data_loader_val = data_loader_val['single']

    if args.eval:
        return None, data_loader_val, global_labels_val

    # Train loader
    print("prepare train dataset")
    sampler_train = RandomSampler(dataset_train)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train if args.seed > -1 else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        generator= generator if args.seed > -1 else None
    )

    return data_loader_train, data_loader_val, global_labels_val
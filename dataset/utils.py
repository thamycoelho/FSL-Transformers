import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from .episodic_dataset import EpisodeDataset


def get_sets(args):
    if args.dataset == 'places':
        from .places import dataset_setting
    elif args.dataset == 'places_600':
        from .places600 import dataset_setting
    else:
        raise ValueError(f'{dataset} is not supported.')

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


def get_loaders(args, num_tasks, global_rank):
    # datasets
    if args.eval:
        _, _, dataset_vals = get_sets(args)
    else:
        dataset_train, dataset_vals, _ = get_sets(args)
    
    val_map_labels = dataset_vals.mapCls
    # Val loader
    # NOTE: meta-dataset has separate val-set per domain
    if not isinstance(dataset_vals, dict):
        dataset_vals = {'single': dataset_vals}

    data_loader_val = {}

    for j, (source, dataset_val) in enumerate(dataset_vals.items()):
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        data_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=3, # more workers can take too much CPU
            pin_memory=args.pin_mem,
            drop_last=False,
            generator=generator
        )
        data_loader_val[source] = data_loader

    if 'single' in dataset_vals:
        data_loader_val = data_loader_val['single']

    if args.eval:
        return None, data_loader_val, val_map_labels

    # Train loader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        generator=generator
    )

    return data_loader_train, data_loader_val, val_map_labels
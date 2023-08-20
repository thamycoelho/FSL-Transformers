#!/bin/bash
device=$1
dataset=$2
experiment=$3
data_path_support=$4
data_path_query=$5


# Extract features
CUDA_VISIBLE_DEVICES=$device python3 extract_features.py --output output/ --dataset $dataset --backbone deit_small --project-name tests --experiment_name $experiment/support --batch-size 50 --extract_features --dataset_path $data_path_support
CUDA_VISIBLE_DEVICES=$device python3 extract_features.py --output output/ --dataset $dataset --backbone deit_small --project-name tests --experiment_name $experiment/query --batch-size 50 --extract_features --dataset_path $data_path_query

CUDA_VISIBLE_DEVICES=$device python3 classify_samples.py --output output/ --dataset $dataset --backbone deit_small --project-name tests --experiment_name $experiment --batch-size 50 --nEpisode 200 --nClsEpisode 50 --classify --support_file output/$dataset/tests/$experiment/support/features.pth --query_file output/$dataset/tests/$experiment/query/features.pth


#!/bin/bash
device=$1
dataset=$2
experiment=$3
data_path_support=$4
data_path_query=$5


# Extract features
CUDA_VISIBLE_DEVICES=$device python3 extract_features.py --output output/dataset_features --dataset $dataset --backbone deit_small --project-name final_test --experiment_name $experiment --batch-size 256 --extract_features --dataset_path $data_path_support 
# CUDA_VISIBLE_DEVICES=$device python3 extract_features.py --output output/ --dataset $dataset --backbone deit_small --project-name final_test --experiment_name $experiment/query --batch-size 50 --extract_features --dataset_path $data_path_query --resume output/places_600/deit/best.pth

# CUDA_VISIBLE_DEVICES=$device python3 classify_samples.py --output output/ --dataset $dataset --backbone deit_small --project-name final_test --experiment_name $experiment --batch-size 256 --nEpisode 10000 --nClsEpisode 8 --classify --support_file output/$dataset/final_test/experiment2/support/features.pth --query_file output/$dataset/final_test/$experiment/query/features.pth

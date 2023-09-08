#!/bin/bash
dataset_path=$1

# Extract features
python3 ../extract_features.py --output ../output/ --dataset csam --backbone deit_small --dataset_path $dataset_path --resume ../output/weights/deit_small/best.pth

# Experiment fsl
echo "CSAM Indoor Experiment 1"
python3 ../classify_samples.py --output ../output/ --dataset csam --experiment_name fsl_protocol --support_file ../output/dataset_features/csam/features.pth --nEpisode 10000 --eval_fsl
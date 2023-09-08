#!/bin/bash
dataset_path=$1
csv_file=$2
csv_sep=$3


# Extract features
python3 ../extract_features.py --output ../output/ --dataset csam_indoor --backbone deit_small --dataset_path $dataset_path --csv_file $csv_file --csv_sep $csv_sep --resume ../output/weights/deit_small/best.pth

# Experiment fsl
echo "CSAM Indoor Experiment 1"
python3 ../classify_samples.py --output ../output/ --dataset csam_indoor --experiment_name fsl_protocol --support_file ../output/dataset_features/csam_indoor/features.pth --nEpisode 10000 --eval_fsl

echo "CSAM Indoor Experiment 2"
python3 ../classify_samples.py --output ../output/ --dataset csam_indoor --experiment_name general_protocol --support_file ../output/dataset_features/places_600_support/features.pth --query_file ../output/dataset_features/csam_indoor/features.pth --nEpisode 10000 --eval_general --batch-size 256

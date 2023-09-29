#!/bin/bash

# To run: ./csam_indoor.sh {path to the folder containing the images of the dataset} {csv file containing the name of the file and the class} {csv separator} {device}
# Example: ./experiment_scripts/csam_indoor.sh ../csam_indoor/files/ cpu

dataset_path=$1
device=$2


# Extract features
python3 extract_features.py --output output/ --dataset csam_indoor --backbone deit_small --dataset_path $dataset_path --csv_file output/csv/scenes_labels.csv --resume output/weights/deit_small/best.pth --device $device

# Experiment fsl
echo "CSAM Indoor Experiment 1"
python3 classify_samples.py --output output/ --dataset csam_indoor --experiment_name fsl_protocol --support_file output/dataset_features/csam_indoor/features.pth --nEpisode 10000 --eval_fsl --device $device

echo "CSAM Indoor Experiment 2"
python3 classify_samples.py --output output/ --dataset csam_indoor --experiment_name general_protocol --support_file output/dataset_features/places_600_support/features.pth --query_file output/dataset_features/csam_indoor/features.pth --nEpisode 10000 --eval_general --batch-size 256 --device $device

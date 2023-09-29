#!/bin/bash

# To run: ./rcpd.sh {path to the folder containing the images of the dataset} {csv file containing the name of the file and the class} {csv separator} {device}
# Example: ./experiment_scripts/rcpd.sh ../RCPD/files/ cpu

dataset_path=$1
device=$2


# Extract features
python3 extract_features.py --output output/ --dataset RCPD --backbone deit_small --dataset_path $dataset_path --csv_file output/csv/rcpd2.csv --resume output/weights/deit_small/best.pth --device $device

# Experiment fsl
echo "RCPD Experiment 1"
python3 classify_samples.py --output output/ --dataset RCPD --experiment_name fsl_protocol --support_file output/dataset_features/RCPD/features.pth --nEpisode 10000 --eval_fsl --device $device
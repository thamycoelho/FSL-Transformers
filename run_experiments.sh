#!/bin/bash
device=$1

for((i=2; i <= $#; i++))
do
backbone=${!i}
echo "experiments for $backbone" 
# train and eval deit
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone --eval
mv output/places_600/$backbone/confusion_matrix.png output/places_600/$backbone/confusion_matrix_no_finetuning.png
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone --eval --resume output/places_600/$backbone/best.pth
mv output/places_600/$backbone/confusion_matrix.png output/places_600/$backbone/confusion_matrix_finetuning.png
done

#!/bin/bash
device=$1

for backbone in deit dino resnet50 resnet50_dino
do
# train and eval deit
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone --eval
CUDA_VISIBLE_DEVICES=$device python3 main.py --output output/ --dataset places_600 --backbone $backbone --experiment_name $backbone --eval --resume output/places_600/$backbone/best.pth
done


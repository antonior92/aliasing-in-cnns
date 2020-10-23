#!/bin/bash

# Resnet 34 - IMAGENET
TIPE=layerwise_aliasing_info
PATH_TO_DATASET=$1
EXTRA=""
MODEL=resnet34;
python evaluate.py --model $MODEL --folder mdl_$MODEL --attack no_attack --dataset imagenet --path_to_dataset $PATH_TO_DATASET --save $TIPE $EXTRA

# Resnet 20 - classfying oscilations
PATH_TO_SCRIPT=${PWD}/../classifying-oscillations
FOLDER=classif-oscil
export PYTHONPATH=${PYTHONPATH}:${PATH_TO_SCRIPT}
python ${PATH_TO_SCRIPT}/train.py python train.py --arch resnet20_small_incr --noise_intens 1.0 --epochs 50 --n_freq 20 --milestones 35 --dropout 0 --save $FOLDER
python evaluate --folder $FOLDER --dataset classif-oscil --save layerwise_aliasing_info

# Generate pie plots plot
python generate_pie_plot.py --path mdl_resnet34/layerwise_aliasing_info.pt --save img/pie-imagenet-resnet34.png
python generate_pie_plot.py --path classif-oscil/layerwise_aliasing_info.pt --type resnet20-classif-oscil --save img/pie-classif-oscil.png
python generate_boxplots.py --folder mdl_resnet34/ --save img/correct-vs-incorrect.png
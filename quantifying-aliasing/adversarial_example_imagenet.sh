#!/bin/bash

# Run experiment
PATH_TO_DATASET=$1
OUTFOLDER=results/pgd_resnet34
MODEL=resnet34;
python evaluate.py --path_to_dataset $PATH_TO_DATASET --folder $OUTFOLDER --attack pgd \
 --eps 0 0.0001 0.0005 0.001 0.005 0.01 0.05 --save layerwise_aliasing_info --model $MODEL
# Plot results
python generate_adversarial_example_plot.py --folder $OUTFOLDER --save img/adversarial_accuracy.png
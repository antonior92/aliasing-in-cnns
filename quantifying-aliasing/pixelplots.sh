#!/bin/bash

IMAGENET_VAL=$1
IMG_DIR=img
INTERM_FOLDER=./results/restricted_evaluation
python evaluate.py --model resnet34 --folder $INTERM_FOLDER --attack no_attack --dataset imagenet --path_to_dataset $IMAGENET_VAL --save dft_folds --start_sample 5287 --max_samples 8
python generate_plots.py --path $INTERM_FOLDER/dft_folds.pt --type 2dplot_pixels  --threshold_reference max --threshold 20  --sample 0 --layer 1_conv1_1 --channel 0 --save $IMG_DIR/aliasing_i.png --hide_cbar
python generate_plots.py --path $INTERM_FOLDER//dft_folds.pt --type 2dplot_pixels  --threshold_reference max --threshold 20  --sample 0 --layer 1_layer2_0_conv1_1 --channel 0 --save $IMG_DIR/aliasing_ii.png --hide_cbar

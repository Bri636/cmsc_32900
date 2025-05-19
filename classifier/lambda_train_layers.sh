#!/bin/bash

# layer train configs
EPOCHS=15
# non-parallel 
python train.py --config_file_path layer_study/epoch_n_layers_2.yaml --epochs $EPOCHS --study layer > ./outputs/layer_study/train/logs/epoch_${EPOCHS}_layers_2.out
python train.py --config_file_path layer_study/epoch_n_layers_4.yaml --epochs $EPOCHS --study layer > ./outputs/layer_study/train/logs/epoch_${EPOCHS}_layers_4.out
python train.py --config_file_path layer_study/epoch_n_layers_8.yaml --epochs $EPOCHS --study layer > ./outputs/layer_study/train/logs/epoch_${EPOCHS}_layers_8.out
python train.py --config_file_path layer_study/epoch_n_layers_16.yaml --epochs $EPOCHS --study layer > ./outputs/layer_study/train/logs/epoch_${EPOCHS}_layers_16.out
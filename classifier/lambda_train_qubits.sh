#!/bin/bash

EPOCHS=15

python train.py --config_file_path qubit_study/epoch_n_qubits_1.yaml --epochs $EPOCHS --study qubit > ./outputs/qubit_study/train/logs/epoch_${EPOCHS}_qubits_1.out
python train.py --config_file_path qubit_study/epoch_n_qubits_2.yaml --epochs $EPOCHS --study qubit > ./outputs/qubit_study/train/logs/epoch_${EPOCHS}_qubits_2.out
python train.py --config_file_path qubit_study/epoch_n_qubits_4.yaml --epochs $EPOCHS --study qubit > ./outputs/qubit_study/train/logs/epoch_${EPOCHS}_qubits_4.out
python train.py --config_file_path qubit_study/epoch_n_qubits_8.yaml --epochs $EPOCHS --study qubit > ./outputs/qubit_study/train/logs/epoch_${EPOCHS}_qubits_8.out
""" Good """

from __future__ import annotations
from datasets import prepare_dataloaders
from model import HybridQNetwork
from utils import get_device, set_seed, get_config, write_cirquit_qubits, write_cirquit_depth
from train import TrainConfig, ModelConfig
from argparse import ArgumentParser, Namespace
from pathlib import Path
from mashumaro.mixins.yaml import DataClassYAMLMixin
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pennylane as qml
from tqdm import tqdm
from typing import Literal
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection, MetricTracker
from matplotlib import pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(args: Namespace,
                   config: TrainConfig,
                   epoch: int,
                   model: HybridQNetwork,
                   test_tracker: MetricTracker,
                   test_cm: ConfusionMatrix,
                   test_loader: DataLoader,
                   device: Literal['cuda', 'mps', 'cpu'],
                   plot_ax: plt.Axes, 
                   ) -> None:
    print(f'TESTING...')
    print(f'============================================================')
    model.eval()
    total_correct = 0.0
    for tidx, tbatch in tqdm(enumerate(test_loader),
                             total=len(test_loader),
                             desc='Testing On Test Set...'):
        test_tracker.increment()

        imgs, labels = tbatch
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        num_correct = torch.sum(preds == labels)
        total_correct += num_correct.item()

        test_cm.update(preds, labels)
        test_tracker.update(preds, labels)
        tqdm.write(
            f'Testing: {total_correct} Correct Out of {len(test_loader) * config.batch_size}...')
        tqdm.write(
            f'Testing Accuracy: {(total_correct / (len(test_loader)*config.batch_size)) * 100}%...')
    print(f'============================================================')

    test_results = test_tracker.compute_all()
    if args.study == 'layer':
        run_name = f'epoch_{epoch}_{config.model_config.num_quantum_layers}_layers'
        save_path_num_layers = Path(
            f'./outputs/layer_study/test/epoch_{epoch}_layers_{config.model_config.num_quantum_layers}.parquet')
        write_cirquit_depth(test_results,
                            None,
                            config.model_config.num_quantum_layers,
                            run_name,
                            save_path_num_layers)
    elif args.study == 'qubit':
        run_name = f'epoch_{epoch}_{config.model_config.num_qubits}_qubits'
        save_path_num_qubits = Path(
            f'./outputs/qubit_study/test/epoch_{epoch}_qubits_{config.model_config.num_qubits}.parquet')
        write_cirquit_qubits(test_results,
                             None,
                             config.model_config.num_qubits,
                             run_name,
                             save_path_num_qubits)
    
    class_names = [str(i) for i in range(config.model_config.num_classes)]
    test_cm.plot(labels=class_names, cmap='magma',
                 ax=plot_ax)
    count = config.model_config.num_qubits if args.study=='qubit' else config.model_config.num_quantum_layers
    plot_ax.set_title(f'Confusion Matrix - {args.study.capitalize()}s per VQC: {count}')
    plot_ax.set_xlabel('Predicted Class')
    plot_ax.set_ylabel('True Class')


def main():

    argparser = ArgumentParser()
    argparser.add_argument('--num_epochs', type=int)
    argparser.add_argument('--study', choices=['layer', 'qubit'])
    args = argparser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    dir_path = f'./outputs/{args.study}_study/train/weights/'
    _ckpt_paths = list(filter(lambda x: int(
        x.split('_')[1]) == args.num_epochs, os.listdir(dir_path)))
    ckpt_paths = [f'./outputs/{args.study}_study/train/weights/' + ckpt_path
                  for ckpt_path in _ckpt_paths]
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f'LOADING MODEL FROM HERE: {ckpt_path}...')
        ckpt = torch.load(ckpt_path)
        _config = ckpt['config']
        config = TrainConfig.from_dict(_config)
        _epoch = ckpt['epoch']

        device = get_device()
        # if config.set_cpu == True:
        #     device = 'cpu'
        set_seed(config.seed, device)
        print(f'Running on Device: {device} With Seed: {config.seed}...')

        _train_loader, _val_loader, test_loader = prepare_dataloaders(config.batch_size,
                                                                      config.val_ratio,
                                                                      config.subset_size)

        model = HybridQNetwork(**asdict(config.model_config)).to(device)
        model.load_state_dict(ckpt['model_state_dict'])

        num_classes = config.num_classes

        test_acc_cls = Accuracy(
            task='multiclass', num_classes=num_classes).to(device)
        test_cm_cls = ConfusionMatrix(
            task='multiclass', num_classes=num_classes).to(device)
        test_metrics = MetricCollection({
            'accuracy': test_acc_cls,
            'confusion': test_cm_cls
        })
        test_tracker = MetricTracker(test_metrics, maximize=True)
        evaluate_model(args,
                       config,
                       args.num_epochs,
                       model,
                       test_tracker,
                       test_cm_cls,
                       test_loader,
                       device, 
                       axes[i])

        del model
        torch.cuda.empty_cache()
        print(f'DELETED MODEL CKPT FROM HERE: {ckpt_path}...')
        
    fig.suptitle(
        f"Confusion Matrices for {args.study.title()} Scaling - Training Epochs: {args.num_epochs}",
        fontsize=16
    )                                              
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    outdir = Path(f'./outputs/{args.study}_study/test/')
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'cm_grid_epoch_{args.num_epochs}.png', dpi=300)
    plt.close(fig)
    print(f"\nSaved combined grid to {outdir}/cm_grid_epoch_{args.num_epochs}.png")

if __name__ == "__main__":

    main()

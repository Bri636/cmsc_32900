from __future__ import annotations
from datasets import prepare_dataloaders
from model import HybridQNetwork, display_model
from utils import save_ckpt, get_device, set_seed, get_config, write_cirquit_depth, write_cirquit_qubits
from argparse import ArgumentParser
from pathlib import Path
from mashumaro.mixins.yaml import DataClassYAMLMixin
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from torch import nn, optim
import pennylane as qml
from tqdm import tqdm
from typing import Literal
from torchmetrics import Accuracy, ConfusionMatrix, MetricCollection, MetricTracker
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelConfig(DataClassYAMLMixin):
    num_classes: int = 10
    num_qubits: int = 4
    num_quantum_layers: int = 6
    embed_size: int = 128

@dataclass
class TrainConfig(DataClassYAMLMixin):
    seed: int = 42
    set_cpu: bool = True
    batch_size: int = 64
    val_ratio: float = 0.2
    subset_size: dict[str, int] = field(
        default_factory=lambda: {'train': 20000,
                                 'test': 1000})
    num_epochs: int = 3
    lr: float = 1e-4
    num_classes: int = 10
    ckpt_every_n: int = 2
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig())


def main():

    argparse = ArgumentParser()
    argparse.add_argument('--config_file_path', default='train_config.yaml')
    argparse.add_argument('--study', choices=['layer', 'qubit'])
    argparse.add_argument('--epochs', type=int)
    args = argparse.parse_args()

    if args.study not in ('layer', 'qubit'): 
        raise ValueError('You can only do two study types')

    yaml_fp = './configs/' + args.config_file_path
    config: TrainConfig = get_config(yaml_fp, TrainConfig)
    config.num_epochs = args.epochs
    device = get_device()
    # if config.set_cpu==True:
    #     device = 'cpu'

    set_seed(config.seed, device)

    print(f'Running on Device: {device} With Seed: {config.seed}...')
    print(f'Performing {args.study} study...')
    train_loader, val_loader, test_loader = prepare_dataloaders(config.batch_size,
                                                                config.val_ratio,
                                                                config.subset_size)
    model = HybridQNetwork(**asdict(config.model_config)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    display_model(model, args.study)

    num_classes = config.num_classes

    val_acc_cls = Accuracy(
        task='multiclass', num_classes=num_classes).to(device)
    val_cm_cls = ConfusionMatrix(
        task='multiclass', num_classes=num_classes).to(device)
    val_metrics = MetricCollection({
        'accuracy': val_acc_cls,
        'confusion': val_cm_cls
    })
    val_tracker = MetricTracker(val_metrics, maximize=True)

    epoch_losses = []
    for epoch in range(1, config.num_epochs + 1):
        print(f'STARTING EPOCH: {epoch} ...')
        print(f'============================================================')

        val_tracker.increment()
        model.train()
        total_loss = 0
        for idx, batch in tqdm(enumerate(train_loader),
                               total=len(train_loader),
                               desc=f'Epoch {epoch}'):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (idx + 1) % 50 == 0:
                micro_avg_loss = (total_loss / (idx + 1))
                tqdm.write(f'Average Loss For 100th Step: {micro_avg_loss}')

        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        model.eval()
        total_correct = 0.0
        for v_idx, vbatch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Validation...'):
            imgs, labels = vbatch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            num_correct = torch.sum((labels == preds))
            total_correct += num_correct.item()

            val_tracker.update(preds, labels)

        tqdm.write(
            f'Validation: {total_correct} Correct Out of {len(val_loader) * config.batch_size}...')
        tqdm.write(
            f'Validation Accuracy: {(total_correct / (len(val_loader)*config.batch_size)) * 100}%...')

        print(f'============================================================')
        
    if args.study=='layer':
        save_path = f'./outputs/{args.study}_study/train/weights/epoch_{epoch}_layers_{config.model_config.num_quantum_layers}.pt'
    elif args.study=='qubit':
        save_path = f'./outputs/{args.study}_study/train/weights/epoch_{epoch}_qubits_{config.model_config.num_qubits}.pt'
    tqdm.write(f'Saving CKPT In Epoch: {epoch} to {save_path}...')
    save_ckpt(model, optimizer, loss, epoch, config, save_path)

    val_results = val_tracker.compute_all()

    if args.study=='layer':
        run_name = f'epoch_{epoch}_{config.model_config.num_quantum_layers}_layers'
        save_path_num_layers = Path(
            f'./outputs/layer_study/train/epoch_{epoch}_layers_{config.model_config.num_quantum_layers}.parquet')
        write_cirquit_depth(val_results,
                            epoch_losses,
                            config.model_config.num_quantum_layers,
                            run_name,
                            save_path_num_layers)
    elif args.study=='qubit': 
        run_name = f'epoch_{epoch}_{config.model_config.num_qubits}_qubits'
        save_path_num_qubits = Path(
            f'./outputs/qubit_study/train/epoch_{epoch}_qubits_{config.model_config.num_qubits}.parquet')
        write_cirquit_qubits(val_results,
                             epoch_losses,
                             config.model_config.num_qubits,
                             run_name,
                             save_path_num_qubits)


if __name__ == "__main__":

    main()

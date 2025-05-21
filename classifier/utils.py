import torch
from torch import nn
from matplotlib import pyplot as plt
import torchmetrics as tm
from torchmetrics.classification import MulticlassConfusionMatrix, Accuracy
from dataclasses import asdict
import random
import numpy as np
from typing import Literal, TypedDict, Protocol, ClassVar, Any, Optional
import pandas as pd
from pathlib import Path
from model import HybridQNetwork
from mashumaro.mixins.yaml import DataClassYAMLMixin

class IsDataclass(Protocol):
    """
    Protocol for general dataclass; just for type signatures
    Source: https://stackoverflow.com/questions/54668000/type-hint-for-an-instance-of-a-non-specific-dataclass
    """
    __dataclass_fields__: ClassVar[dict[str, Any]] 
 
# Pennylane MPS not supported yet
def get_device() -> Literal['cuda', 'mps', 'cpu']:
    if torch.cuda.is_available(): 
        device = 'cuda'
    else: 
        device = 'cpu' 
    return device

def set_seed(seed: int = 42, device: Literal['cuda', 'mps', 'cpu'] = 'mps'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device=='cuda': 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device=='mps': 
        torch.mps.manual_seed(seed)

def save_ckpt(model: nn.Module, 
              optimizer: torch.optim.Optimizer, 
              loss: float, 
              epoch: int, 
              config: IsDataclass, 
              save_path: str
              ) -> None: 
    ckpt = {
        'epoch': epoch, 
        'loss': loss, 
        'config': asdict(config), 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(ckpt, save_path)


def get_config(path: Path | str, default_config_cls: DataClassYAMLMixin) -> DataClassYAMLMixin: 
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'============================================================')
    if not path.exists(): 
        print(f'No Config Detected at {path}, Making One...')
        _config = default_config_cls()
        with open(path, "w", encoding="utf-8") as f:
            f.write(_config.to_yaml())
        config = _config
        
    else: 
        print(f'Reading Config From {path}...')
        with open(path, "r", encoding="utf-8") as f:
            _config = f.read()  
        config = default_config_cls.from_yaml(_config)
        
    print(f'Config:\n=======\n{_config}')
    print(f'============================================================')
    return config


def write_cirquit_depth(results: dict,
                        epoch_losses: Optional[list],
                        circuit_depth: int,
                        name: str,
                        log_file_path: str) -> None:
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    accuracies = [acc.item() for acc in results['accuracy']]
    epochs = list(range(1, len(accuracies)+1))
    
    df_run = pd.DataFrame({
        'run': name,
        'circuit_depth': circuit_depth,
        'epochs': epochs,
        'accuracy': accuracies,
        'loss': epoch_losses
    })
    
    df_all = df_run
    df_all.to_parquet(log_file_path, index=False)
    print(f'Saving or Overwriting to Path: {log_file_path}...')

def write_cirquit_qubits(results: dict,
                         epoch_losses: list,
                         num_qubits: int,
                         name: str,
                         log_file_path: str) -> None:
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    accuracies = [acc.item() for acc in results['accuracy']]
    epochs = list(range(1, len(accuracies)+1))

    df_run = pd.DataFrame({
        'run': name,
        'num_qubits': num_qubits,
        'epochs': epochs,
        'accuracy': accuracies, 
        'loss': epoch_losses
    })
    df_all = df_run
    df_all.to_parquet(log_file_path, index=False)
    print(f'Saving or Overwriting to Path: {log_file_path}...')
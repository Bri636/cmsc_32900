""" GOOD """

from __future__ import annotations

from typing import Sequence, Callable, Optional
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split

def prepare_dataloaders(batch_size: int = 64, 
                        val_frac: float = 0.2,
                        subset_size: Optional[dict] = None) -> Sequence[DataLoader]: 
    train_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Lambda(lambda image: image.convert('RGB')),
        torchvision.transforms.AugMix(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    _train_dataset = datasets.FashionMNIST(root=".", train=True, transform=train_transform, download=True)
    test_dataset = datasets.FashionMNIST(root=".", train=False, transform=train_transform, download=True)

    if subset_size: 
        train_subset_size = subset_size['train']
        test_subset_size = subset_size['test']
        assert len(_train_dataset) >= train_subset_size, f'''
        Train Dataset {len(_train_dataset)} must be larger than {train_subset_size} 
        '''
        assert len(test_dataset) >= test_subset_size, f'''
        Test Dataset {len(test_dataset)} must be larger than {test_subset_size}
        '''
        
        train_indices = [idx for idx in range(train_subset_size)]
        _train_dataset = Subset(_train_dataset, indices=train_indices)
        
        test_indices = [idx for idx in range(test_subset_size)]
        test_dataset = Subset(test_dataset, indices=test_indices)

    num_total = len(_train_dataset)
    num_val   = int(num_total * val_frac)
    num_train = num_total - num_val
    train_dataset, validation_dataset = random_split(_train_dataset, [num_train, num_val])

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True)
    validation_loader = DataLoader(dataset=validation_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             drop_last=True)
    
    return train_loader, validation_loader, test_loader
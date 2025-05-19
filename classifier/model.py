""" Good """

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import pennylane as qml 
from pennylane.measurements import ExpectationMP
from pennylane.qnn.torch import TorchLayer
from qiskit import QuantumCircuit

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import torch
from torch import nn
from torchvision import models

from dataclasses import dataclass, field, asdict
from typing import Literal
import warnings
warnings.filterwarnings("ignore")

def display_model(model: HybridQNetwork, study: Literal['layer', 'qubit']) -> None:
    fig = model.draw_first_layer()
    plt.title(f'VQC Layer With {model.num_qubits} Qubits and {model.num_qlayers} Layers')
    fig.savefig(f'./outputs/{study}_study/images/layers_{model.num_qlayers}_qubits_{model.num_qubits}.png')

def quantum_layer(num_qubits: int, num_layers: int, device: str) -> TorchLayer: 
    """ Adapted from Pennylane: https://pennylane.ai/qml/demos/tutorial_qnn_module_torch """
    dev = qml.device('default.qubit', wires=num_qubits)
    
    @qml.qnode(dev, interface='torch')
    def quantum_node(inputs, weights) -> list[ExpectationMP]:
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
        qml.BasicEntanglerLayers(weights, wires=range(num_qubits), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
    
    weights = {'weights': (num_layers, num_qubits)}
    return TorchLayer(quantum_node, weights)

class HybridQNetwork(nn.Module): 
    """ Hybrid layer that interleaves Resnet18, VQC layer, and classifier """
    def __init__(self, 
                 num_classes: int = 10, 
                 num_qubits: int = 2, 
                 num_quantum_layers: int = 6, 
                 embed_size: int = 128,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        device = next(self.backbone.parameters()).device
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if embed_size % num_qubits != 0: 
            raise ValueError(f'Embed Size {embed_size} must be divisible by {num_qubits} qubits')
        
        num_qubit_layers = embed_size // num_qubits
        qubit_layers = []
        for _ in range(num_qubit_layers): 
            qubit_layers.append(quantum_layer(num_qubits, num_quantum_layers, device))
            
        self.qubit_layers = nn.ModuleList(qubit_layers)
        self.num_qubits = num_qubits
        self.num_qlayers = num_quantum_layers
                
        self.proj = nn.Linear(512, embed_size)
        
        self.proj2 = nn.Linear(embed_size, embed_size)
        self.act2 = nn.ReLU()
        self.proj3 = nn.Linear(embed_size, embed_size)
        self.act3 = nn.ReLU()
        self.classifier = nn.Linear(embed_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.proj(x)
        x_split: tuple[torch.Tensor] = torch.split(x, self.num_qubits, dim=1)
        q_outs: list[torch.Tensor] = [q_layer(_x) for q_layer, _x in 
                                      zip(self.qubit_layers, x_split)]
        x = torch.cat(q_outs, axis=1)
        x = self.proj2(x)
        x = self.act2(x)
        x = self.proj3(x)
        x = self.act3(x)
        out = self.classifier(x)
        return out
    
    def draw_first_layer(self) -> Figure: 
        qnode_1 = self.qubit_layers[0].qnode
        qnode_1(torch.rand(self.num_qubits), 
                torch.rand((self.num_qlayers, self.num_qubits)))
        
        qasm_str = qnode_1._tape.to_openqasm()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        fig: Figure = qc.draw(output='mpl')
        print(qc.draw())
        return fig

       
       
       
       
        

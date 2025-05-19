""" GOOD """
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
import numpy as np
import os
from argparse import ArgumentParser

from utils import get_config
from train import TrainConfig

np.random.seed(42)

def entangling_capability(num_qubits: int,
                          num_layers: int,
                          num_samples: int = 1000,
                          seed: int | None = None) -> float:
    """
    Since Pennylane does not support meyer-wallach measure yet, I use community implementation
    
    Source: https://discuss.pennylane.ai/t/meyer-wallach-measure/3763/3 
    """
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(num_qubits), rotation=qml.RY)
        return qml.state()

    q_values = []
    for _ in range(num_samples):
        inputs = np.random.uniform(0, 2.0 * np.pi, num_qubits)
        weights = np.random.uniform(0, 2.0 * np.pi, (num_layers, num_qubits))
        state = circuit(inputs, weights)
        psi = state.reshape([2] * num_qubits)

        sum_purity = 0.0
        for q in range(num_qubits):
            psi = np.moveaxis(psi, q, 0).reshape(2, -1)
            rho = psi @ psi.conj().T
            purity = np.trace(rho @ rho).real
            sum_purity += purity
        # Meyer–Wallach Q
        Q = 2.0 * (1.0 - purity_sum / num_qubits)
        q_values.append(Q)

    return float(np.mean(q_values))

argparser = ArgumentParser()
argparser.add_argument('--study', choices=['qubit','layer'])
args = argparser.parse_args()
config_dir = f'./configs/{args.study}_study/'

config_paths = os.listdir(config_dir)
for path in config_paths: 
    yaml_fp = config_dir + path
    config: TrainConfig = get_config(yaml_fp, TrainConfig)
    
    num_qubits = config.model_config.num_qubits
    num_layers = config.model_config.num_quantum_layers
    num_samples = 1024
    seed = 42
    
    Q_est = entangling_capability(num_qubits, num_layers, num_samples, seed)
    print(f"Estimated Entangling Capability Q = {Q_est:.3f}\n")
    print(f'============================================================')

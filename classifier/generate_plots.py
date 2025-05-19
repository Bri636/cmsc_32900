""" Good """

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import matplotlib.ticker as mticker

from typing import Literal

def merge_circuit_data(dir_path: str, study: Literal['layer', 'qubit'], epochs: int) -> pd.DataFrame: 
    """ Merges all .parquet files inside a dir """
    print(f'READING IN FROM DIR: {dir_path}')
    dir_path = Path(dir_path)
    if not dir_path.exists(): 
        raise FileNotFoundError(f'File Directory Does Not Exist, Data is Being Written Elsewhere...')
    
    file_pattern = f'epoch_{epochs}_*.parquet'
    pq_files: list[str] = sorted(dir_path.rglob(file_pattern))
    dfs = [pd.read_parquet(pq_file) for pq_file in pq_files]
    df = pd.concat(dfs, ignore_index=True)
    
    if study=='layer': 
        df.sort_values(["circuit_depth", "run", "epochs"], inplace=True)
    elif study=='qubit': 
        df.sort_values(["num_qubits", "run", "epochs"], inplace=True)
    
    return df

def plot_layer_study(df: pd.DataFrame) -> Figure:
    depths = np.sort(df["circuit_depth"].unique())
    cmap   = cm.get_cmap("viridis", len(depths))
    depth_to_color = {d: cmap(i) for i, d in enumerate(depths)}

    fig_acc, ax = plt.subplots(figsize=(8, 5))
    for run_name, grp in df.groupby("run"):
        depth = grp["circuit_depth"].iloc[0]
        ax.plot(grp["epochs"],
                grp["accuracy"] * 100,
                marker="o",
                label=f"Layers: {depth}",
                color=depth_to_color[depth])
    ax.set_ylim(0, 100)  
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy") 
    ax.set_title("Validation Accuracy vs Epoch by VQC Layer Depth")
    ax.grid(alpha=.3)
    ax.legend(title='Number of Layers')
    
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda i: int(labels[i].split()[-1])
    )
    ax.legend([handles[i] for i in order],
            [labels[i]  for i in order],
            title="Number of Layers")
    
    fig_loss, ax_loss = plt.subplots(figsize=(8,5))
    for run, grp in df.groupby("run"):
        d = grp["circuit_depth"].iloc[0]
        ax_loss.plot(grp["epochs"], grp["loss"],
                     marker="o", 
                     label=f"Layers: {d}", 
                     color=depth_to_color[d])
    ax_loss.set(title="Training Loss vs Epoch by VQC Layer Depth", xlabel="Epoch", ylabel="Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(title="Number of Layers")

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    
    handles, labels = ax_loss.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split()[-1]))
    ax_loss.legend([handles[i] for i in order],
                [labels[i]  for i in order],
                title="Number of Layers") 
    return fig_acc, fig_loss
        
        
def plot_qubit_study(df: pd.DataFrame) -> Figure:
    depths = np.sort(df["num_qubits"].unique())
    cmap   = cm.get_cmap("viridis", len(depths))
    depth_to_color = {d: cmap(i) for i, d in enumerate(depths)}

    fig_acc, ax = plt.subplots(figsize=(8, 5))
    for run_name, grp in df.groupby("run"):
        depth = grp["num_qubits"].iloc[0]
        ax.plot(grp["epochs"],
                grp["accuracy"] * 100,
                marker="o",
                label=f"Qubits: {depth}",
                color=depth_to_color[depth])
    ax.set_ylim(0, 100) 
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy %") 
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy vs Epoch by Qubits per VQC")
    ax.grid(alpha=.3)
    ax.legend(title='Number of Qubits')
    
    fig_loss, ax_loss = plt.subplots(figsize=(8,5))
    for run, grp in df.groupby("run"):
        q = grp["num_qubits"].iloc[0]
        ax_loss.plot(grp["epochs"], grp["loss"],
                     marker="o", 
                     label=f"Qubits: {q}", 
                     color=depth_to_color[q])
    ax_loss.set(title="Training Loss vs Epoch by Qubits per VQC", xlabel="Epoch", ylabel="Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(title="Number of Qubits")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    return fig_acc, fig_loss 
        
        
def plot_train_study(df: pd.DataFrame, study: Literal['layer', 'qubit']) -> tuple[Figure, ...]: 
    """ Interface for generating plots for layer or qubit study"""
    if study=='layer': 
        fig_acc, fig_loss = plot_layer_study(df)
    elif study=='qubit': 
        fig_acc, fig_loss = plot_qubit_study(df)
    return fig_acc, fig_loss
        

def plot_accuracy_vs_depth(df: pd.DataFrame) -> Figure:
    final_acc = (
        df
        .sort_values(["run", "circuit_depth", "epochs"])
        .groupby(["run", "circuit_depth"], as_index=False)
        .last()
    )
    summary = (
        final_acc
        .groupby("circuit_depth")["accuracy"]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    summary["mean"] *= 100                            
    summary["std"]  *= 100  
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary["circuit_depth"].astype(str)
    y = summary["mean"]
    yerr = summary["std"]
    
    ax.bar(x, y, yerr=yerr, capsize=5)
    _annotate_bars(ax, x, y)   
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy vs VQC Layer Depth")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    return fig

def _annotate_bars(ax, x, y):
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9) 

def plot_accuracy_vs_qubits(df: pd.DataFrame) -> Figure:
    final_acc = (
        df
        .sort_values(["run", "num_qubits", "epochs"])
        .groupby(["run", "num_qubits"], as_index=False)
        .last()
    )
    summary = (
        final_acc
        .groupby("num_qubits")["accuracy"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    
    summary["mean"] *= 100                            
    summary["std"]  *= 100  
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary["num_qubits"].astype(str)
    y = summary["mean"]
    yerr = summary["std"]
    
    ax.bar(x, y, yerr=yerr, capsize=5)
    _annotate_bars(ax, x, y)   
    ax.set_xlabel("Number Qubits")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy vs Qubits per VQC")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    return fig

def plot_test_study(df: pd.DataFrame, study: Literal['layer', 'qubit']) -> Figure: 
    if study=='layer': 
        fig = plot_accuracy_vs_depth(df)
    elif study=='qubit': 
        fig = plot_accuracy_vs_qubits(df)
    return fig
    
def main(): 
    
    argparser = ArgumentParser()
    argparser.add_argument('--study', type=str, choices=['layer', 'qubit'])
    argparser.add_argument('--type', type=str, choices=['train', 'test'])
    argparser.add_argument('--epochs', type=int)
    args = argparser.parse_args()
    
    assert args.study in ('layer', 'qubit'), f'''Study must be layer or qubit'''
    assert args.type in ('train', 'test'), f'''You must choose train or test'''
    
    dir_path = f'./outputs/{args.study}_study/{args.type}/'
    df = merge_circuit_data(dir_path, args.study, args.epochs)
    
    if args.type=='train':
        fig_acc, fig_loss = plot_train_study(df, args.study)
        acc_out_path = dir_path + f'ep_{args.epochs}_{args.type}_acc_{args.study}_study_plot.png'
        loss_out_path = dir_path + f'ep_{args.epochs}_{args.type}_loss_{args.study}_study_plot.png'
        
        Path(acc_out_path).parent.mkdir(exist_ok=True, parents=True)
        fig_acc.tight_layout()
        fig_acc.savefig(acc_out_path)
        print(f"{args.study}‑study acc figure saved at {acc_out_path}")
        
        Path(loss_out_path).parent.mkdir(exist_ok=True, parents=True)
        fig_loss.tight_layout()
        fig_loss.savefig(loss_out_path)
        print(f"{args.study}‑study loss figure saved at {loss_out_path}")
      
    elif args.type=='test': 
        acc_out_path = dir_path + f'ep_{args.epochs}_{args.type}_acc_{args.study}_study_plot.png'
        fig_acc = plot_test_study(df, args.study)
        Path(acc_out_path).parent.mkdir(exist_ok=True, parents=True)
        fig_acc.tight_layout()
        fig_acc.savefig(acc_out_path)
        print(f"{args.study}‑study acc figure saved at {acc_out_path}")
    
if __name__=="__main__": 
    
    main()

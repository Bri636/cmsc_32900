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


def plot_train_study(df: pd.DataFrame, study: Literal['layer', 'qubit']) -> Figure:

    if study=='layer': 
        col = 'circuit_depth'
        plt_label = "Layers: {metric}"
        acc_title = "Validation Accuracy vs Epoch by VQC Layer Depth"
        loss_title = "Training Loss vs Epoch by VQC Layer Depth"
        legend_title = 'Number of Layers'
        
    elif study=='qubit': 
        col = 'num_qubits'
        plt_label = "Qubits: {metric}"
        acc_title = "Validation Accuracy vs Epoch by Qubits per VQC"
        loss_title = "Training Loss vs Epoch by Qubits per VQC"
        legend_title = 'Number of Qubits'
    
    depths = np.sort(df[col].unique())
    cmap   = cm.get_cmap("viridis", len(depths))
    depth_to_color = {d: cmap(i) for i, d in enumerate(depths)}

    fig_acc, ax = plt.subplots(figsize=(8, 5))
    for _run_name, grp in df.groupby("run"):
        metric = grp[col].iloc[0]
        ax.plot(grp["epochs"],
                grp["accuracy"] * 100,
                marker="o",
                label=plt_label.format(metric=metric),
                color=depth_to_color[metric])
        
    ax.set_ylim(0, 100)  
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy") 
    ax.set_title(acc_title)
    ax.grid(alpha=.3)
    ax.legend(title=legend_title)
    
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: int(labels[i].split()[-1]))
    ax.legend([handles[i] for i in order],
              [labels[i]  for i in order],
              title=legend_title)
    
    fig_loss, ax_loss = plt.subplots(figsize=(8,5))
    for run, grp in df.groupby("run"):
        metric = grp[col].iloc[0]
        ax_loss.plot(grp["epochs"], grp["loss"],
                     marker="o", 
                     label=plt_label.format(metric=metric), 
                     color=depth_to_color[metric])
    ax_loss.set(title=loss_title, xlabel="Epoch", ylabel="Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(title=legend_title)
    ax_loss.legend([handles[i] for i in order],
                   [labels[i]  for i in order],
                   title=legend_title) 

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    
    return fig_acc, fig_loss
        
        
def plot_test_study(df: pd.DataFrame, study: Literal['layer', 'qubit']): 
    
    if study=='layer': 
        col = 'circuit_depth'
        xlab = "Circuit Depth"
        title = "Test Accuracy vs VQC Layer Depth"
    elif study=='qubit': 
        col = 'num_qubits'
        xlab = "Number Qubits"
        title = "Test Accuracy vs Qubits per VQC"
    
    def annotate_bars(ax, x, y):
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 1, f"{yi:.1f}%", 
                    ha="center", va="bottom", fontsize=9) 
        
    final_acc = (df.sort_values(["run", col, "epochs"])
                 .groupby(["run", col], as_index=False).last())
    summary = (final_acc.groupby(col)["accuracy"]
               .agg(mean="mean", std="std")
               .reset_index())

    summary["mean"] *= 100                            
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary[col].astype(str)
    y = summary["mean"]
    
    ax.bar(x, y, capsize=5)
    annotate_bars(ax, x, y)   
    ax.set_xlabel(xlab)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    
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
        print(f"{args.study} study acc figure saved at {acc_out_path}")
        
        Path(loss_out_path).parent.mkdir(exist_ok=True, parents=True)
        fig_loss.tight_layout()
        fig_loss.savefig(loss_out_path)
        print(f"{args.study} study loss figure saved at {loss_out_path}")
      
    elif args.type=='test': 
        acc_out_path = dir_path + f'ep_{args.epochs}_{args.type}_acc_{args.study}_study_plot.png'
        fig_acc = plot_test_study(df, args.study)
        Path(acc_out_path).parent.mkdir(exist_ok=True, parents=True)
        fig_acc.tight_layout()
        fig_acc.savefig(acc_out_path)
        print(f"{args.study} study acc figure saved at {acc_out_path}")
    
if __name__=="__main__": 
    
    main()

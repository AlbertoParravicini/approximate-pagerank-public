#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:30:36 2020

@author: aparravi
"""

import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os
import matplotlib.lines as lines
import pandas as pd
import numpy as np
import scipy.stats as st
from matplotlib.patches import Patch, Rectangle
from plot_exec_time import get_exp_label, get_upper_ci_size
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D


# Define some colors;
c1 = "#b1494a"
c2 = "#256482"
c3 = "#2f9c5a"
c4 = "#28464f"
c5 = "#FFEA70"

r4 = "#CE1922"
r3 = "#F41922"
r2 = "#FA3A51"
r1 = "#FA4D4A"
r5 = "#F07B71"
r6 = "#F0A694"

b1 = "#97E6DB"
b2 = "#C6E6DB"
b3 = "#CEF0E4"
b4 = "#9CCFC4"

b5 = "#AEDBF2"
b6 = "#B0E6DB"
b7 = "#B6FCDA"
b8 = "#7bd490"

bt1 = "#55819E"
bt2 = "#538F6F"

MAIN_RESULT_FOLDER = "../../../../data/results/raw_results/2020_01_25"
DATE = "2020_05_142"

def kendall_tau(reference_rank, predicted_rank):
    
    # Items with correct relative rank;
    c_plus = 0
    # Items without correct relative rank;
    c_minus = 0
    # Items for which a ranking exists in the predicted rank;
    c_s = 0
    # Items for which a ranking exists in the reference rank;
    c_u = 0
    
    item_set = set(reference_rank + predicted_rank)
    reference_rank_dict = {item: pos for pos, item in enumerate(reference_rank)}
    predicted_rank_dict = {item: pos for pos, item in enumerate(predicted_rank)}
    
    for i, item_1 in enumerate(item_set):
        for j, item_2 in enumerate(item_set):
            # Consider each pair exactly once;
            if i >= j:
                continue
            else:
                ref_found = False
                pred_found = False
                if item_1 in reference_rank_dict and item_2 in reference_rank_dict:
                    ref_found = True
                    c_u += 1
                if item_1 in predicted_rank_dict and item_2 in predicted_rank_dict:
                    pred_found = True
                    c_s += 1
                if ref_found and pred_found:
                    if (reference_rank_dict[item_1] - reference_rank_dict[item_2]) * (predicted_rank_dict[item_1] - predicted_rank_dict[item_2]) > 0:
                        c_plus += 1
                    else:
                        c_minus += 1
                        
    return (c_plus - c_minus) / (np.sqrt(c_u) * np.sqrt(c_s))


def read_data():
    
    result_list = []
    
    for graph_name in os.listdir(MAIN_RESULT_FOLDER):
        dir_name = os.path.join(MAIN_RESULT_FOLDER, graph_name)
        if os.path.isdir(dir_name):
            for res_path in os.listdir(dir_name):
                res_file = os.path.join(MAIN_RESULT_FOLDER, graph_name, res_path)
                if res_file.endswith(".csv"):
                    with open(res_file, "r") as f:
                        # Read results, but skip the header;
                        result = f.readlines()[1]
                        
                        # Parse the file name;
                        try:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, _, n_iter = res_path[:-4].split("-")
                        except ValueError:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, n_iter = res_path[:-4].split("-")
                        max_iter = max_iter.replace("it", "")
                        # Parse the result line;
                        skip_extra = False
                        try:
                            _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist, convergence_error, mae, fpga_predictions, cpu_predictions = result.split(",")
                        except ValueError:
                            _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist = result.split(",")
                            convergence_error = ""
                            mae = ""
                            fpga_predictions = ""
                            cpu_predictions = ""
                            skip_extra = True
                        
                        n_ppr = int(n_ppr)
                        if n_ppr == 8:
                        
                            # Obtain the single error metrics;
                            errors = errors.split(";")
                            ndcg = ndcg.split(";")
                            edit_dist = edit_dist.split(";")
                            mae = mae.split(";")
                            errors_10 = [int(e.split("|")[0]) for e in errors if e.strip()]
                            errors_20 = [int(e.split("|")[1]) for e in errors if e.strip()]
                            errors_50 = [int(e.split("|")[2]) for e in errors if e.strip()]
                            ndcg_10 = [float(e.split("|")[0]) for e in ndcg if e.strip()]
                            ndcg_20 = [float(e.split("|")[1]) for e in ndcg if e.strip()]
                            ndcg_50 = [float(e.split("|")[2]) for e in ndcg if e.strip()]
                            edit_dist_10 = [int(e.split("|")[0]) for e in edit_dist if e.strip()]
                            edit_dist_20 = [int(e.split("|")[1]) for e in edit_dist if e.strip()]
                            edit_dist_50 = [int(e.split("|")[2]) for e in edit_dist if e.strip()]
                            
                            if not skip_extra:
                                mae_10 = [float(e.split("|")[0]) for e in mae if e.strip()]
                                mae_20 = [float(e.split("|")[1]) for e in mae if e.strip()]
                                mae_50 = [float(e.split("|")[2]) for e in mae if e.strip()]
                                
                                cpu_predictions = cpu_predictions.split(";")
                                fpga_predictions = fpga_predictions.split(";")
                                
                                cpu_predictions_10 = [[int(x) for x in e.split("|")[:10]] for e in cpu_predictions if e.strip()]
                                cpu_predictions_20 = [[int(x) for x in e.split("|")[:20]] for e in cpu_predictions if e.strip()]
                                cpu_predictions_50 = [[int(x) for x in e.split("|")[:50]] for e in cpu_predictions if e.strip()]
                                fpga_predictions_10 = [[int(x) for x in e.split("|")[:10]] for e in fpga_predictions if e.strip()]
                                fpga_predictions_20 = [[int(x) for x in e.split("|")[:20]] for e in fpga_predictions if e.strip()]
                                fpga_predictions_50 = [[int(x) for x in e.split("|")[:50]] for e in fpga_predictions if e.strip()]
                                
                                prec_10 = [len(set(c).intersection(set(f))) / 10 for (c, f) in zip(cpu_predictions_10, fpga_predictions_10)]
                                prec_20 = [len(set(c).intersection(set(f))) / 20 for (c, f) in zip(cpu_predictions_20, fpga_predictions_20)]
                                prec_50 =[len(set(c).intersection(set(f))) / 50 for (c, f) in zip(cpu_predictions_50, fpga_predictions_50)]
                                
                                kendall_10 = [kendall_tau(c, f) for (c, f) in zip(cpu_predictions_10, fpga_predictions_10)]
                                kendall_20 = [kendall_tau(c, f) for (c, f) in zip(cpu_predictions_20, fpga_predictions_20)]
                                kendall_50 =[kendall_tau(c, f) for (c, f) in zip(cpu_predictions_50, fpga_predictions_50)]
                            else:
                                mae_10 = [0] * n_ppr
                                mae_20 = [0] * n_ppr
                                mae_50 = [0] * n_ppr
                                prec_10 = [0] * n_ppr
                                prec_20 = [0] * n_ppr
                                prec_50 = [0] * n_ppr
                                kendall_10 = [0] * n_ppr
                                kendall_20 = [0] * n_ppr
                                kendall_50 = [0] * n_ppr
                            
                            assert(n_ppr == len(errors_10))
                            assert(n_ppr == len(errors_20))
                            assert(n_ppr == len(errors_50))
                            assert(n_ppr == len(ndcg_10))
                            assert(n_ppr == len(ndcg_20))
                            assert(n_ppr == len(ndcg_50))
                            assert(n_ppr == len(edit_dist_10))
                            assert(n_ppr == len(edit_dist_20))
                            assert(n_ppr == len(edit_dist_50))
                            
                            # Add the result line to the list;
                            for i in range(n_ppr):  
                                new_res_line = [graph_name, int(V), int(E), n_bit, int(n_ppr),
                                                1, int(max_iter), float(exec_time_ms), float(transfer_time_ms),
                                                errors_10[i], errors_20[i], errors_50[i],
                                                ndcg_10[i], ndcg_20[i], ndcg_50[i],
                                                edit_dist_10[i], edit_dist_20[i], edit_dist_50[i],
                                                convergence_error,
                                                mae_10[i], mae_20[i], mae_50[i],
                                                prec_10[i], prec_20[i], prec_50[i],
                                                kendall_10[i], kendall_20[i], kendall_50[i]]
                                result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter", 
                                      "exec_time_ms", "transfer_time_ms",
                                      "errors_10", "errors_20", "errors_50", 
                                      "ndcg_10", "ndcg_20", "ndcg_50",
                                      "edit-dist_10", "edit-dist_20", "edit-dist_50",
                                      "convergence_error",
                                      "mae_10", "mae_20", "mae_50",
                                      "prec_10", "prec_20", "prec_50",
                                      "kendall_10", "kendall_20", "kendall_50"])
    
    res_agg = result_df.groupby(["graph_name", "V", "E", "n_bit"], as_index=False).mean()
    return result_df, res_agg


if __name__ == "__main__":
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 40 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    res, agg = read_data()
    
    # Skip float results;
    res = res[res["n_bit"] != "float"]
    
    # Consider only graphs with 100k vertices;
    res = res[res["V"] == 2 * 10**5]
    # res = res[~res["V"].isin([10**5, 2 * 10**5])]
    
    #%%
    
    # Setup plot;
    graph_names = ["$\mathdefault{G_{n,p}}$", "Watts–Strogatz", "Holme and Kim"]
    error_metrics = ["Num. Errors", "Edit Distance", "NDCG (higher is better)"]
    error_metrics_raw = ["errors", "edit-dist", "ndcg"]
    error_max = [50, 50, 1]
    error_min = [0, 0, 0.9]
    error_sizes = [10, 20, 50]
    num_col = len(res["graph_name"].unique())
    num_rows = len(error_metrics) 
    fig = plt.figure(figsize=(2.0 * num_col, 2.8 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.80,
                    bottom=0.10,
                    left=0.18,
                    right=0.95,
                    hspace=0.6,
                    wspace=0.6)

    markers = ["o", "X", "D"]
        
    # One row per graph;
    for i, group in enumerate(res.groupby(["graph_name"])):
        data = group[1]
        data = data.melt(id_vars=["n_bit"], value_vars=[e + "_" + str(d) for e in error_metrics_raw for d in error_sizes])
        data["error_type"] = [s.split("_")[0] for s in data["variable"]]
        data["error_size"] = [int(s.split("_")[1]) for s in data["variable"]]
        
        # One column per error metric;
        for j, e in enumerate(error_metrics_raw):
            
            curr_data = data[data["error_type"] == e]
            ax = fig.add_subplot(gs[j, i])
            ax = sns.lineplot(x="n_bit", y="value", hue="error_size", data=curr_data, palette=[r1, b8, b2], ax=ax,
                  err_style="bars", linewidth=3, legend=False, zorder=2, ci=None)
            data_averaged = curr_data.groupby(["n_bit", "error_size"], as_index=False).mean()
            ax = sns.scatterplot(x="n_bit", y="value", hue="error_size", data=data_averaged, palette=[r1, b8, b2], ax=ax, edgecolor="#0f0f0f",
                  size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="error_size", linewidth=0.05)
            ax.set_ylim([error_min[j], error_max[j]])
            ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
            ax.set_xlabel(None)
            if i == 0:
                ax.set_ylabel(f"{error_metrics[j]}", fontsize=16)
            else:
                ax.set_ylabel(None)
            # if i == 0:
            #      # Graph name;
            ax.annotate(f"{graph_names[i]}",
                        xy=(0.5, 1), xycoords="axes fraction", fontsize=14, textcoords="offset points", xytext=(0, 15),
                        horizontalalignment="center", verticalalignment="center")
            # ax.set_title(f"{graph_names[i]}", fontsize=14, loc="center", xytext=(-40, 25))
            
            # Set the number of ticks on the y axis;
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

               
            if j == 2:
                ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right")
                
            # Turn off tick lines;
            ax.xaxis.grid(False)  
            sns.despine(ax=ax)              
            ax.tick_params(labelcolor="black", labelsize=12, pad=6)
            
              
            
    plt.annotate("Fixed-point Bitwidth", fontsize=18, xy=(0.5, 0.015), xycoords="figure fraction", ha="center")
           
    fig.suptitle(f"PPR Accuracy w.r.t\nfixed-point bitwidth,\n|V|= {get_exp_label(2 * 10**5)}, |E|=~{get_exp_label(2 * 10**6)}",
                 fontsize=18, ha="left", x=0.05)
    
    # Legend;
    custom_lines = [
        Line2D([], [], color="white", marker=markers[0],
               markersize=10, label="Top-10", markerfacecolor=r1, markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[1],
               markersize=10, label="Top-20", markerfacecolor=b8, markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[2],
               markersize=10, label="Top-50", markerfacecolor=b2, markeredgecolor="#2f2f2f"),
        ]
    
    leg = fig.legend(custom_lines, ["Top-10", "Top-20", "Top-50"],
                             bbox_to_anchor=(0.98, 1), fontsize=16)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/errors_{DATE}_large.pdf")
    
    
    #%% Same plot, but aggregate over all graphs;

    res, agg = read_data()
    
    res = res[res["n_bit"] != "float"]
    
    #%%
    
    plt.rcParams['mathtext.fontset'] = "cm" 
    
    # Setup plot;
    error_metrics = ["Num. Errors", "Edit Distance", "NDCG\n(higher is better)", "MAE", "Precision\n(higher is better)", "Kendall's " + r"$\mathbf{\tau}$" + "\n(higher is better)"]
    error_metrics_raw = ["errors", "edit-dist", "ndcg", "mae", "prec", "kendall"]
    error_max = [50, 50, 1, 0.12, 1, 1]
    error_min = [0, 0, 0.9, 0, 0.7, 0.6]
    error_sizes = [10, 20, 50]
    num_rows = 2
    num_col = len(error_metrics) // num_rows

    fig = plt.figure(figsize=(2.0 * num_col, 3.1 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.75,
                    bottom=0.12,
                    left=0.12,
                    right=0.95,
                    hspace=0.8,
                    wspace=0.6)

    data = res.melt(id_vars=["n_bit"], value_vars=[e + "_" + str(d) for e in error_metrics_raw for d in error_sizes])
    data["error_type"] = [s.split("_")[0] for s in data["variable"]]
    data["error_size"] = [int(s.split("_")[1]) for s in data["variable"]]
    
    markers = ["o", "X", "D"]
    
    # One column per error metric;
    for j, e in enumerate(error_metrics_raw):
        
        curr_data = data[data["error_type"] == e]
        ax = fig.add_subplot(gs[j // num_col, j % num_col])
        ax = sns.lineplot(x="n_bit", y="value", hue="error_size", data=curr_data, palette=[r1, b8, b2], ax=ax,
              err_style="bars", linewidth=3, legend=False, zorder=2, ci=None, estimator="mean")
        data_averaged = curr_data.groupby(["n_bit", "error_size"], as_index=False).mean()
        ax = sns.scatterplot(x="n_bit", y="value", hue="error_size", data=data_averaged, palette=[r1, b8, b2], ax=ax, edgecolor="#0f0f0f",
              size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="error_size", linewidth=0.05)
        ax.set_ylim([error_min[j], error_max[j]])
        ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
       
        ax.annotate(f"{error_metrics[j]}",
                    xy=(0.5, 1), xycoords="axes fraction", fontsize=12, textcoords="offset points", xytext=(0, 20),
                    horizontalalignment="center", verticalalignment="center")
        # ax.set_title(f"{graph_names[i]}", fontsize=14, loc="center", xytext=(-40, 25))
        
        # Set the number of ticks on the y axis;
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

           
        if j == 2:
            ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=12)
            
        # Turn off tick lines;
        ax.xaxis.grid(False)  
        sns.despine(ax=ax)              
        ax.tick_params(labelcolor="black", labelsize=12, pad=6)
            
              
            
    plt.annotate("Fixed-point Bitwidth", fontsize=16, xy=(0.5, 0.015), xycoords="figure fraction", ha="center")
           
    fig.suptitle(f"PPR Accuracy w.r.t\nfixed-point bitwidth,\naggregated over all graphs",
                 fontsize=18, ha="left", x=0.05)
    
    # Legend;    
    custom_lines = [
        Line2D([], [], color="white", marker=markers[0],
               markersize=10, label="Top-10", markerfacecolor=r1, markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[1],
               markersize=10, label="Top-20", markerfacecolor=b8, markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[2],
               markersize=10, label="Top-50", markerfacecolor=b2, markeredgecolor="#2f2f2f"),
        ]
    
    leg = fig.legend(custom_lines, ["Top-10", "Top-20", "Top-50"],
                             bbox_to_anchor=(0.98, 1), fontsize=12)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/errors_{DATE}_agg.pdf")           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
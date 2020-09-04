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

MAIN_RESULT_FOLDER = "../../../../data/results/raw_results/2020_02_04"
DATE = "2020_02_04"

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
                        _, _, n_bit, n_ppr, max_iter, _, _, _, _, _, n_iter = res_path[:-4].split("-")
                        max_iter = max_iter.replace("it", "")
                        # Parse the result line;
                        _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist = result.split(",")
                        
                        n_ppr = int(n_ppr)
                        
                        # Obtain the single error metrics;
                        errors = errors.split(";")
                        ndcg = ndcg.split(";")
                        edit_dist = edit_dist.split(";")
                        errors_10 = [int(e.split("|")[0]) for e in errors if e.strip()]
                        errors_20 = [int(e.split("|")[1]) for e in errors if e.strip()]
                        errors_50 = [int(e.split("|")[2]) for e in errors if e.strip()]
                        ndcg_10 = [float(e.split("|")[0]) for e in ndcg if e.strip()]
                        ndcg_20 = [float(e.split("|")[1]) for e in ndcg if e.strip()]
                        ndcg_50 = [float(e.split("|")[2]) for e in ndcg if e.strip()]
                        edit_dist_10 = [int(e.split("|")[0]) for e in edit_dist if e.strip()]
                        edit_dist_20 = [int(e.split("|")[1]) for e in edit_dist if e.strip()]
                        edit_dist_50 = [int(e.split("|")[2]) for e in edit_dist if e.strip()]
                        
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
                                            edit_dist_10[i], edit_dist_20[i], edit_dist_50[i],]
                            result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter", 
                                      "exec_time_ms", "transfer_time_ms",
                                      "errors_10", "errors_20", "errors_50", 
                                      "ndcg_10", "ndcg_20", "ndcg_50",
                                      "edit-dist_10", "edit-dist_20", "edit-dist_50"])
    
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
    
    # Consider only graphs with 100k vertices;
    res = res[res["V"] == 2 * 10**5]
    
    # Setup plot;
    graph_names = ["$\mathdefault{G_{n,p}}$", "Watts–Strogatz", "Holme and Kim"]
    error_metrics = ["Num. Errors", "Edit Distance", "NDCG"]
    error_metrics_raw = ["errors", "edit-dist", "ndcg"]
    error_max = [50, 50, 1]
    error_min = [0, 0, 0.9]
    error_sizes = [10, 20, 50]
    num_col = len(error_metrics) 
    num_rows = len(res["graph_name"].unique())
    fig = plt.figure(figsize=(2.0 * num_col, 4.0 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.77,
                    bottom=0.08,
                    left=0.13,
                    right=0.95,
                    hspace=0.5,
                    wspace=0.5)
    
    # One row per graph;
    for i, group in enumerate(res.groupby(["graph_name"])):
        data = group[1]
        data = data.melt(id_vars=["n_bit"], value_vars=[e + "_" + str(d) for e in error_metrics_raw for d in error_sizes])
        data["error_type"] = [s.split("_")[0] for s in data["variable"]]
        data["error_size"] = [int(s.split("_")[1]) for s in data["variable"]]
        
        # One column per error metric;
        for j, e in enumerate(error_metrics_raw):
            
            curr_data = data[data["error_type"] == e]
            ax = fig.add_subplot(gs[i, j])
            ax = sns.lineplot(x="n_bit", y="value", hue="error_size", data=curr_data, palette=[r1, b8, b2], ax=ax,
                  err_style="bars", linewidth=2, legend=False, zorder=2, ci=None)
            data_averaged = curr_data.groupby(["n_bit", "error_size"], as_index=False).mean()
            ax = sns.scatterplot(x="n_bit", y="value", hue="error_size", data=data_averaged, palette=[r1, b8, b2], ax=ax, edgecolor="#2f2f2f",
                  size_norm=40, legend=False, zorder=3, ci=None, markers=["X", "X", "X"], style="error_size", linewidth=0.1)
            ax.set_ylim([error_min[j], error_max[j]])
            ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
            ax.set_xlabel(None)
            if j > 0:
                ax.set_ylabel(None)
            else:
                ax.set_ylabel("Errors")
            if i == 0:
                ax.set_title(f"{error_metrics[j]}", fontsize=18, loc="center")
            if j == 0:
                # Graph name;
                ax.annotate(f"{graph_names[i]}",
                            xy=(0, 1), xycoords="axes fraction", fontsize=16, textcoords="offset points", xytext=(-40, 25),
                            horizontalalignment="left", verticalalignment="center")
            if j == 2:
                ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=14)
                
            # for c_i, c in enumerate(sorted(data["n_bit"].unique())):
            #     rectangles = []
            #     segments = []
            #     for s_i, s in enumerate(error_sizes):
            #         temp_data = curr_data[(curr_data["error_size"] == s) & (curr_data["n_bit"] == c)]
            #         y = np.mean(temp_data["value"])
            #         width = 0.1
            #         height = get_upper_ci_size(temp_data["value"], 0.95 if s_i < 2 else 0.5) * 2
            #         lower_left = [c_i - width / 2, y - height / 2]
            #         # Add an offset to the x position, to avoid overlapping;
            #         lower_left[0] += s_i * (width / 2) - width / 2
            #         rectangles += [Rectangle(lower_left, width, height)]
                    
            #         segments += [[
            #                 (lower_left[0] + width / 2, y - np.std(temp_data["value"])),
            #                 (lower_left[0] + width / 2, y + np.std(temp_data["value"]))
            #                 ]]
        
            #     pc = PatchCollection(rectangles, facecolor=[r1, b8, b2], edgecolor="#2f2f2f", linewidth=0.2, zorder=3, clip_on=False)        
            #     lc = LineCollection(segments, linewidth=0.5, zorder=3, clip_on=True, color="#2f2f2f")
            #     ax.add_collection(lc)    
            #     ax.add_collection(pc)     
                
            # Turn off tick lines;
            ax.xaxis.grid(False)  
            sns.despine(ax=ax)              
            ax.tick_params(labelcolor="black", labelsize=14, pad=7)
            
              
            
    plt.annotate("Fixed-point Bitwidth", fontsize=18, xy=(0.5, 0.025), xycoords="figure fraction", ha="center")
           
    fig.suptitle(f"PPR Accuracy w.r.t fixed-point bitwidth,\n|V|= {get_exp_label(2 * 10**5)}, |E|=~{get_exp_label(2 * 10**6)}",
                 fontsize=22, ha="left", x=0.05)
    
    # Legend;
    custom_lines = [
        Patch(facecolor=r1, edgecolor="#2f2f2f", label="Top-10"),
        Patch(facecolor=b8, edgecolor="#2f2f2f", label="Top-20"),
        Patch(facecolor=b2, edgecolor="#2f2f2f", label="Top-50"),
        ]
    
    leg = fig.legend(custom_lines, ["Top-10", "Top-20", "Top-50"],
                             bbox_to_anchor=(0.98, 0.95), fontsize=16)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/errors_{DATE}_large.pdf")
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
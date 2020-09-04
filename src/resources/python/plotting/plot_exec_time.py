#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:52:11 2020

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
import matplotlib.ticker as ticker
from matplotlib import transforms


# Define some colors;
c1 = "#b1494a"
c2 = "#256482"
c3 = "#2f9c5a"
c4 = "#28464f"

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

bb0 = "#FFA685"
bb1 = "#75B0A2"
bb2 = b3
bb3 = b7
bb4 = "#7ED7B8"
bb5 = "#7BD490"

bt1 = "#55819E"
bt2 = "#538F6F"

MAIN_RESULT_FOLDER = "../../../../data/results/raw_results/2020_05_15"
CPU_RESULT_FOLDER = "../../../../data/results/summary/pagerank/cpu_opt/2020_03_11"
DATE = "2020_05_28"


def get_exp_label(val):
    
    # Get the power of 10
    exp_val = 0
    remaining_val = val
    while (remaining_val % 10 == 0):
        exp_val += 1
        remaining_val = remaining_val // 10
    if remaining_val > 1:
        return r"$\mathdefault{" + str(remaining_val) + r"·{10}^" + str(exp_val) + r"}$"
    else:
        return r"$\mathdefault{" + r"{10}^" + str(exp_val) + r"}$"
    
    
def add_labels(ax, labels=None, vertical_offsets=None, skip_zero=True, patch_num=None, fontsize=14, rotation=0, 
               format_str="{:.2f}x", label_color="#2f2f2f"):
    
    if not vertical_offsets:
        vertical_offsets = [ax.get_ylim()[1] * 0.05] * len(ax.patches)
    if not labels:
        labels = [p.get_height() for p in ax.patches]
    patches = []
    if not patch_num:
        patches = ax.patches
    else:
        patches = [p for i, p in enumerate(ax.patches) if i in patch_num]
            
    # Iterate through the list of axes' patches
    lab_num = 0
    for p in patches:
        if labels[lab_num] and (lab_num > 0 or not skip_zero):
            ax.text(p.get_x() + p.get_width()/2., vertical_offsets[lab_num] + p.get_height(), format_str.format(labels[lab_num]), 
                    fontsize=fontsize, color=label_color, ha='center', va='bottom', rotation=rotation)
        lab_num += 1
        

# Compute the size of the upper 0.95 interval, i.e. the size between the top of the bar and the top of the error bar;
def get_upper_ci_size(x, ci=0.95):
    ci_upper = st.t.interval(ci, len(x)-1, loc=np.mean(x), scale=st.sem(x))[1]
    return ci_upper - np.mean(x)


def read_datasets():
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
                        if int(n_ppr) == 8:
                            max_iter = max_iter.replace("it", "")
                            # Parse the result line;
                            try:
                                _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist, convergence_error, rmse, fpga_predictions, cpu_predictions = result.split(",")
                            except ValueError:
                                _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist = result.split(",")
                                convergence_error = ""
                                rmse = ""
                                fpga_predictions = ""
                                cpu_predictions = ""
                                
                            # Add the result line to the list;
                            new_res_line = [graph_name, int(V), int(E), n_bit, int(n_ppr),
                                            1, int(max_iter), float(exec_time_ms),
                                            float(transfer_time_ms), errors, ndcg, edit_dist,
                                            convergence_error, rmse, fpga_predictions, cpu_predictions]
                            result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter", 
                                      "exec_time_ms", "transfer_time_ms",
                                      "errors", "ndcg", "edit_dist",
                                      "convergence_error", "rmse", "fpga_predictions", "cpu_predictions"])
        
    res_grouped = result_df.groupby(["graph_name", "V", "E", "n_bit", "n_ppr"], as_index=False).sum()
    
    # Amortize the execution time over 100 vertices;
    res_grouped["exec_time_ms"] = 100 * res_grouped["exec_time_ms"] / (res_grouped["n_ppr"] * res_grouped["n_iter"])
    
    # Same stuff, for CPU results;
    cpu_result_list = []
    
    for res_name in os.listdir(CPU_RESULT_FOLDER):
        if (res_name.endswith(".csv")):
            res_path = os.path.join(CPU_RESULT_FOLDER, res_name)
            with open(res_path, "r") as f:
                graph_name = res_name.split("_")[6]
                # Skip header;
                for l in f.readlines()[1:]:
                    _, n_iter, _, _, _, V, E, _, exec_time_ms, _, _, _ = l.split(",")
                    
                    # Add the result line to the list;
                    new_res_line = [graph_name, int(V), int(E), "cpu", 1,
                                    1, 10, float(exec_time_ms),
                                    0, 0, 0, 0]
                    cpu_result_list += [new_res_line]
                    
    # Create a dataframe;
    cpu_result_df = pd.DataFrame(cpu_result_list, 
                                 columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter",
                                          "exec_time_ms", "transfer_time_ms",
                                          "errors", "ndcg", "edit_dist"])    
    # Skip the error metrics;
    cpu_res_grouped = cpu_result_df.groupby(["graph_name", "V", "E", "n_bit", "n_ppr"], as_index=False).sum().iloc[:, :-3]  

    # Join the datasets;
    res = pd.concat([res_grouped, cpu_res_grouped])        
    
    # Compute speedup;
    res["speedup"] = res["exec_time_ms"]
    
    # Compute median CPU exec. time;
    cpu_median_time = {}
    for g in res.groupby(["V"]):
        data = g[1]
        cpu_median_time[g[0]] = np.median(data[data["n_bit"] == "cpu"]["exec_time_ms"])
        
    for g in res.groupby(["graph_name", "V", "n_bit"]):
        graph = g[0][0]
        v = g[0][1]
        bit = g[0][2]
        res.loc[(res["graph_name"] == graph) & (res["V"] == v) & (res["n_bit"] == bit), "speedup"] /= cpu_median_time[v]
    res["speedup"] = 1 / res["speedup"]
        
    return res


#%%  
    
if __name__ == "__main__":
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 1 
    
    res = read_datasets()
    
    # Remove gplus graph;
    res = res[res["graph_name"] != "gplus"]
    
    sorted_sizes = [100000, 200000, 128000, 81306]
    title_labels = [r"$\mathdefault{|E|=10^6}$", r"$\mathdefault{|E|=2 · 10^6}$", "Amazon", "Twitter"]
    
    # Set the number of vertices to string, for cat plotting;
    # res["V"] = res["V"].astype(str)
    
    num_row = 2
    num_col = len(res.groupby(["V"])) // num_row  # Assume that graphs have a constant degree;
    # fig = plt.figure(figsize=(2 * num_col, 2.6 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.78,
    #                 bottom=0.15,
    #                 left=0.15,
    #                 right=.99,
    #                 hspace=1,
    #                 wspace=0.35)
    
    fig = plt.figure(figsize=(1.8 * num_col, 2.4 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.85,
                    bottom=0.12,
                    left=0.08,
                    right=.99,
                    hspace=1.2,
                    wspace=0.35)
    
    palettes = [[r1, bb0, bb2, bb3, bb4, bb5]] * num_col * num_row
    
    # fig.suptitle("Execution Time Speedup\nw.r.t CPU Baseline", fontsize=16, x=.05, y=0.99, ha="left")
                       
    # g = sns.catplot(x="n_bit", y="speedup", col="V", sharey=False, data=res, kind="bar", height=3, aspect=0.5,
    #                 alpha=1, edgecolor="#2f2f2f", 
    #                 capsize=.05, errwidth=0.8, zorder=2)
    
    # Remove legend, we add it later on (super hack here, there's no other way to prevent the barplot from drawing a legend);
    # ax.legend().remove()
    
    groups = res.groupby(["V"])
    groups = sorted(groups, key=lambda x: sorted_sizes.index(x[0]))
    
    for i, group in enumerate(groups):
        ax = fig.add_subplot(gs[i // num_row, i % num_row])
        ax0 = ax
        
        # Replace "float" with "32float" to guarantee the right bar sorting;
        group[1].loc[group[1]["n_bit"] == "float", "n_bit"] = "32float"
        
        data = group[1].sort_values(["n_bit"], ascending=False).reset_index(drop=True)
        ax = sns.barplot(x="n_bit", y="speedup", data=data, palette=palettes[i], capsize=.05, errwidth=0.8, ax=ax,
                          edgecolor="#2f2f2f")
        sns.despine(ax=ax)
        # ax.set_title(f"|V|= {get_exp_label(group[0])}, |E|=~{get_exp_label(10 * group[0])}", fontsize=16, loc="left")
        # if i == 0:
        #     ax.set_ylabel("Average Speedup", fontsize=16)  
        # else:
        #     ax.set_ylabel("")  
        # ax.set_xlabel("Fixed-point Bitwidth", fontsize=16) 
        ax.set_ylabel("")
        ax.set_xlabel("")
        labels = ax.get_xticklabels()
        cpu_label = int(np.mean(data[data["n_bit"] == "cpu"]["exec_time_ms"]))
        for j, l in enumerate(labels):
            if j == 0:
                l.set_text(f"CPU")
            elif (j == 1) and len(labels) > 5:
               l.set_text("F32")
                
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', labelsize=8)
        # ax.tick_params(axis='y', which='major', labelsize=10)
        
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}x"))
        ax.tick_params(axis='y', which='major', labelsize=8)
        
        # Speedup labels;
        offsets = []
        for b in data["n_bit"].unique():
            offsets += [get_upper_ci_size(data.loc[data["n_bit"] == b, "speedup"], ci=0.80)]
        offsets = [o if not np.isnan(o) else 0.2 for o in offsets]
        # Manually fix some offsets;
        if i == 0:
            offsets[1] = 0.3
        if i == 1:
            offsets[1] = 0.1
        if i == 3:
            offsets[1] = 0.1
        add_labels(ax, vertical_offsets=offsets, rotation=90, fontsize=10)
        
        # Reference execution time label;
        # add_labels(ax, labels=[cpu_label], skip_zero=False, label_color=r4, 
                    # format_str="CPU: {} ms", patch_num=[0], rotation=90, fontsize=14)
        
        # Add graph type;
        ax.annotate(f"{title_labels[i]}", xy=(0.6, 1), fontsize=12, ha="center", xycoords="axes fraction", xytext=(0.55, 1.35))
                
        ax.annotate(f"CPU Baseline:", xy=(0.5, 0.0), fontsize=9, ha="center", xycoords="axes fraction", xytext=(0.5, -0.28))
        ax.annotate(f"{cpu_label} ms", xy=(0.5, 0.0), fontsize=9, color=r4, ha="center", xycoords="axes fraction", xytext=(0.5, -0.40))
        
    # plt.annotate("Fixed-point Bitwidth", xy=(0.5, 0.03), fontsize=14, ha="center", va="center", xycoords="figure fraction")
    # plt.annotate("Average Speedup", xy=(0.05, 0.5), fontsize=14, ha="center", va="center", rotation=90, xycoords="figure fraction")
                    
    plt.savefig(f"../../../../data/plots/exec_time_{DATE}.pdf")

    
    
    
    
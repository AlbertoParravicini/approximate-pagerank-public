#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:20:51 2019

@author: aparravi
"""

import networkx as nx
import os
import time
import argparse
from datetime import datetime
from scipy.io import mmwrite

OUTPUT_FOLDER = "../../../../data/graphs/mtx"
TYPES = ["pc", "scf", "smw", "gnp"]
DEFAULT_TYPE = TYPES[0]

N = 100
PC_M = 10
PC_P = 0.1
SMW_K = 20
SMW_P = 0.2
GNP_DEGREE = 10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a random graph")
    
    parser.add_argument("-t", "--type", metavar="[pc|scf|smw|gnp]", default=DEFAULT_TYPE,
                        help="Type of graph")
    parser.add_argument("-u", "--undirect_graph", action='store_true',
                        help="If present, undirect the input graph")
    parser.add_argument("-a", "--all_types", action='store_true',
                        help="If present, generate graphs using all the avaliable models")
    parser.add_argument("-c", "--csr", action='store_true',
                        help="If present, store as CSR, else as CSC")
    parser.add_argument("-n", "--vertices", type=int, nargs="+",
                        help="number of vertices of the graph")   
    args = parser.parse_args()
    
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    graph_type = args.type
    if graph_type not in TYPES:
        graph_type = DEFAULT_TYPE
        
    graph_types = [graph_type]
    if args.all_types:
        graph_types = TYPES
    
    undirect = args.undirect_graph
    store_csr = args.csr
    num_vertices = [n for n in args.vertices if n > 0] if len(args.vertices) > 0 else [N]
    print(num_vertices)
    
    # Create the graph;
    for t in graph_types:
        for n in num_vertices:
            print("-" * 30)
            print(f"Generating graph of type {t}...")
            start = time.time()
            if t == "pc":
                print(f"- |N|: {n};")
                g = nx.powerlaw_cluster_graph(n, PC_M, PC_P)
            elif t == "scf":
                print(f"- |N|: {n};")
                g = nx.scale_free_graph(n, alpha=0.05, beta=0.9, gamma=0.05)
            elif t == "smw":
                print(f"- |N|: {n}; K: {SMW_K}; P: {SMW_P};")
                g = nx.watts_strogatz_graph(n, SMW_K, SMW_P)
            elif t == "gnp":
                print(f"- |N|: {n}; sparsity: {GNP_DEGREE / n};")
                g = nx.fast_gnp_random_graph(n, GNP_DEGREE / n, directed=True)
            else:
                print(f"ERROR: invalid graph: {graph_type}!")
                exit(-1)
                
            end = time.time()
            print(f"\nGraph generation time: {(end - start):.2f} seconds")
            v = g.number_of_nodes()
            e = g.number_of_edges()
            print(f"- |N|: {v}; |E|: {e}; sparsity: {e / e**2:.4f}")
            
            # Undirect the graph;
            if undirect:
                print("\nUndirecting graph...")
                start = time.time()
                g = g.to_undirected()
                end = time.time()
                print(f"Graph undirection time: {(end - start):.2f} seconds")
            
            # Save the graph as MTX;
            print(f"\nGenerating sparse matrix from graph...")
            start = time.time()
            matrix = nx.to_scipy_sparse_matrix(g, format="csr" if store_csr else "csc")
            end = time.time()
            print(f"Matrix generation time: {(end - start):.2f} seconds")
            
            print(f"\nStoring sparse matrix...")
            output = os.path.join(OUTPUT_FOLDER, f"{t}_{v}_{e}_{now}.mtx")
            start = time.time()
            matrix = mmwrite(output, matrix)
            end = time.time()
            print(f"Matrix storage time: {(end - start):.2f} seconds")
    
    
    

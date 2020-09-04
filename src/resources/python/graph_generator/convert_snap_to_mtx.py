#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:23:14 2020

@author: aparravi
"""

import argparse
import time
import os

DEFAULT_OUTPUT_FOLDER = "../../../../data/graphs/mtx"
SEPARATOR = " "
SORT = True

# Convert a graph downloaded from SNAP to MTX;
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert a CSC matrix stored in Matrix Market Format to an edgelist graph readable by Pgx")
    
    parser.add_argument("-i", "--input", metavar="<path/to/input/mtx>",
                        help="Path to the file where the mtx is stored")
    parser.add_argument("-o", "--output", metavar="<path/to/output/edgelist>",
                        help="Path to the folder the output is stored")
    parser.add_argument("-m", "--max_vertices", metavar="N", type=int,
                        help="Maximum number of vertices to consider")
    
    args = parser.parse_args()
    
    input_graph = args.input
    output_folder = args.output if args.output else DEFAULT_OUTPUT_FOLDER
    max_vertices = args.max_vertices if args.max_vertices else -1
    
    num_vertices = 0
    num_edges = 0
    
    # If true, swap start and end when writing the output;
    store_as_csc = True
    
    output_graph = os.path.join(output_folder, os.path.splitext(os.path.basename(input_graph))[0] + ".mtx")
    
    print(f"reading {input_graph}")
    print(f"writing to {output_graph}")
    print(f"max vertices={max_vertices}")
    
    # Store tuples of edges;
    edges = set()
    # Store updated ids of vertices;
    id_to_id_map = {}
    
    start_time = time.time()
    
    with open(input_graph) as f:
        for i, l in enumerate(f.readlines()):
            num_edges += 1
            values = l.strip().split(SEPARATOR)
            start, end = [int(x) for x in values]
            
            # Process edge start;
            if start in id_to_id_map:
                start_f = id_to_id_map[start]
            else:
                start_f = num_vertices
                id_to_id_map[start] = num_vertices
                num_vertices += 1
            
            # Process edge end;
            if end in id_to_id_map:
                end_f = id_to_id_map[end]
            else:
                end_f = num_vertices
                id_to_id_map[end] = num_vertices
                num_vertices += 1
                
            if max_vertices > 0 and num_vertices >= max_vertices:
                break
                
            edges.add((start_f, end_f))
            
            if i > 0 and i % 100000 == 0:
                print(f"vertices: {num_vertices}, lines: {num_edges}, time: {(time.time() - start_time):.2f} sec")
                
    # If sort is true, sort edges by start or end;
    if SORT:
        print("sorting...")
        edges = sorted(edges, key=lambda x: (x[0], x[1]))
       
    print(f"storing graph with |V|={num_vertices}, |E|={num_edges}")
    with open(output_graph, "w") as o:
        
        o.write("%%MatrixMarket matrix coordinate integer symmetric\n")
        o.write(f"% filename: {output_graph}\n")
        o.write(f"{num_vertices} {num_vertices} {num_edges}\n")
        
        for s, e in edges:       
            if store_as_csc:
                o.write(f"{e + 1} {s + 1} 1\n")    
            else:
                o.write(f"{s + 1} {e + 1} 1\n")    
    print("done!")
                
                
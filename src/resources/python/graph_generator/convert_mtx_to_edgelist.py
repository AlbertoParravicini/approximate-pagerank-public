#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:06:07 2020

@author: aparravi
"""

import argparse
import time
import os
import json

DEFAULT_OUTPUT_FOLDER = "../../../../data/graphs/edgelist"

# Convert a CSC matrix stored in Matrix Market Format to an edgelist graph readable by Pgx;
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert a CSC matrix stored in Matrix Market Format to an edgelist graph readable by Pgx")
    
    parser.add_argument("-i", "--input", metavar="<path/to/input/mtx>",
                        help="Path to the file where the mtx is stored")
    parser.add_argument("-f", "--input_folder", metavar="<path/to/input/folder>",
                        help="Path to the folder where the mtx are stored")
    parser.add_argument("-o", "--output", metavar="<path/to/output/edgelist>",
                        help="Path to the folder the output is stored")
    
    args = parser.parse_args()
    
    input_graphs = [args.input]
    input_folder = args.input_folder
    output_folder = args.output if args.output else DEFAULT_OUTPUT_FOLDER
    
    if input_folder:
        input_graphs = [os.path.join(input_folder, x) for x in os.listdir(input_folder) if x.endswith(".mtx")]
      
    for input_graph in input_graphs:
        num_vertices = 0
        num_edges = 0
        
        output_graph = os.path.join(output_folder, os.path.splitext(os.path.basename(input_graph))[0] + ".edgelist")
        
        print(f"reading {input_graph}")
        print(f"writing to {output_graph}")
        
        MTX_HEADER_SIZE = 2
        
        start_time = time.time()
        with open(output_graph, "w") as o:
            with open(input_graph) as f:
                for i, l in enumerate(f.readlines()):
                    if i == MTX_HEADER_SIZE:
                        num_vertices, _, num_edges = [int(x) for x in l.strip().split(" ")]
                        print(f"-- |N|: {num_vertices}, |E|: {num_edges}")
                    if i > MTX_HEADER_SIZE:
                        # Replace end with start as we read from a CSC, and ignore the third value;
                        values = l.strip().split(" ")
                        if len(values) == 3:
                            end, start, _ = [int(x) for x in values]
                        elif len(values) == 2:
                             end, start = [int(x) for x in values]
                        else:
                            raise ValueError("unparsable MTX line: {l}")
                        # MTX vertices are 1-indexed, but use 0-indexing in the edgelist;
                        o.write(f"{start - 1} {end - 1}\n")                    
                        
                    if ((i > 0 and i % 100000 == 0) or (i - MTX_HEADER_SIZE == num_edges)):
                        print(f"vertices: {num_vertices}, lines: {i - MTX_HEADER_SIZE} / {num_edges} ({100 * (i - MTX_HEADER_SIZE) / num_edges}%), time: {(time.time() - start_time):.2f} sec")
                
        # Create a Json with the graph configuration;
        output_json = os.path.join(output_folder, os.path.splitext(os.path.basename(input_graph))[0] + ".json")
        
        config_dict = {"uris": [os.path.basename(output_graph)], "format": "edge_list", "vertex_id_type": "long", "separator": " "}
    
        # Write the JSON to output;
        with open(output_json, 'w') as f:
            json.dump(config_dict, f)
    

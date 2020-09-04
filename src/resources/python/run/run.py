#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import argparse
import subprocess
import pandas as pd

##############################
##############################

RESULT_FOLDER = "../../../../data/results/summary/pagerank"
NUM_TESTS = 10
DEFAULT_GRAPH = "../../../../data/graphs/mtx/graph_small_c16.mtx"
MAX_ITER = 100
MAX_ERR = 0.000001
ALPHA = 0.8
DEFAULT_EDGELIST_FOLDER = "../../data/graphs/edgelist/benchmark"

# Path to the executable of each test.

# Protip: create a symlink as "ln -s ~/workspace_2018_2/approximate_pagerank/Emulation-SW Emulation-SW"
TEST_PATHS = {
        "cpu": "../../../common/golden_algorithms/bin/pagerank",
        "cpu_opt": "../../../cpu/gradlew run -q -p ../../../cpu --args=\"-n ppr",
        "fpga": "",
        "fpga_a": "../../../../build/fpga/pagerank",
        "fpga_sw_emu": "",
        "fpga_a_sw_emu": "",
        "gpu": ""        
        }

XCLBIN_PATHS = {
        "fpga": "",
        "fpga_a": "../../../../build/pagerank.xclbin",
        "fpga_sw_emu": "",
        "fpga_a_sw_emu": "",
        }

# Template to run the executables;
CMD = "{} {} -t {} -g {} {} -m {} -e {} -a {} {} {} {}"

##############################
##############################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Approximate PageRank on FPGA")
    
    parser.add_argument("-t", "--test", nargs='+', 
                        help="List of tests to be performed, 1 or more from (cpu|cpu_opt|fpga|fpga_a|fpga_sw_emu|fpga_a_sw_emu|gpu)")    
    parser.add_argument("-d", "--debug", action='store_true',
                        help="If present, print debug messages")
    parser.add_argument("-n", "--num_tests", metavar="N", type=int, default=NUM_TESTS,
                        help="Number of times each test is executed")
    parser.add_argument("-g", "--graph", metavar="path/to/graph", default=DEFAULT_GRAPH,
                       help="Path to a graph stored in MTX format, or to a folder with a graph stored as CSC, or with a folder containing MTX")
    parser.add_argument("-c", "--use_csc", action='store_true',
                        help="If present, the input is assumed to be a folder containing a CSC")
    parser.add_argument("-m", "--max_iter", metavar="N", type=int, default=MAX_ITER,
                        help="Maximum number of iterations of PageRank")
    parser.add_argument("-e", "--max_err", metavar="N", type=float, default=MAX_ERR,
                        help="Maximum accepted error at convergence")
    parser.add_argument("-a", "--dangling_factor", metavar="N", type=float, default=ALPHA,
                        help="Value of dangling factor")
    parser.add_argument("-u", "--undirect_graph", action='store_true',
                        help="If present, undirect the input graph")
    
    args = parser.parse_args()
    
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    debug = args.debug
    num_tests = args.num_tests if args.num_tests > 0 else NUM_TESTS
    use_csc = args.use_csc
    if args.use_csc:
        if os.path.isdir(args.graph):
            graphs = [args.graph]
        else:
            use_csc = False
            graphs = [DEFAULT_GRAPH]
    elif os.path.isfile(args.graph) and args.graph.endswith(".mtx"):
        use_csc = False
        graphs = [args.graph]
    elif os.path.isdir(args.graph):
        graphs = [os.path.join(args.graph, x) for x in os.listdir(args.graph) if x.endswith(".mtx")]
    else:
        use_csc = False
        graphs = [DEFAULT_GRAPH]
    max_iter = args.max_iter if args.max_iter > 0 else MAX_ITER
    max_err = args.max_err if args.max_err >= 0 else MAX_ERR
    alpha = args.dangling_factor if (0 <= args.dangling_factor <= 1) else ALPHA
    undirect = args.undirect_graph
    tests = [t for t in args.test if t in TEST_PATHS]
    
    # Execute each test;
    for graph in graphs:
        for t in tests:
            res = []
            
            xclbin = ""
            if "fpga" in t:
                xclbin = XCLBIN_PATHS[t]
    
            # Pick the equivalent edgelist graph if running "cpu_opt";
            graph_edgelist = graph
            if t == "cpu_opt":
                graph_edgelist = os.path.join(DEFAULT_EDGELIST_FOLDER, os.path.splitext(os.path.basename(graph))[0] + ".json")
                
            cmd = CMD.format(
                    TEST_PATHS[t],
                    "-d" if debug else "",
                    num_tests,
                    graph_edgelist if t == "cpu_opt" else graph,
                    "-c" if use_csc else "",
                    max_iter,
                    max_err,
                    alpha,
                    "-u" if undirect else "",
                    "-x" if xclbin else "",
                    xclbin
                    )
            # Need to add an additional quote to close the gradlew args;
            if t == "cpu_opt":
                cmd += "\""
                
            # Set the environment variable required for software emulation;
            if "sw_emu" in t:
                os.environ["XCL_EMULATION_MODE"] = "sw_emu"
            
            print(f"executing {cmd}")
            output = subprocess.check_output(cmd, shell=True)
            # Parse the result;
            if not debug:
                res += [f"{t},{i},{s}" for i, s in enumerate(output.decode("utf-8").split("\n")) if s]
                # Remove a logging message added by Pgx;
                if t == "cpu_opt":
                    res = res[:-1]
            else:
                out = [s for s in output.decode("utf-8").split("\n")]
                for s in out:
                    print(s)
           
            # Remove the environment variable;
            if "sw_emu" in t:
                os.environ["XCL_EMULATION_MODE"] = ""  
                    
            # Obtain a dataframe with the results;
            if not debug:
                df = pd.DataFrame([r.split(",") for r in res], columns=["type", "iteration",
                                   "fixed_float_width", "fixed_float_scale", "graph", "v", "e", "source", "exec_time",
                                   "transfer_time", "error_pct", "ndcg"])
                print(df)
                graph_base_name = os.path.splitext(os.path.basename(graph))[0]
                output_path = os.path.join(RESULT_FOLDER, f"{now}_{graph_base_name}_{t}.csv")
                if not os.path.exists(RESULT_FOLDER):
                    os.makedirs(RESULT_FOLDER)
                df.to_csv(output_path, index=False)

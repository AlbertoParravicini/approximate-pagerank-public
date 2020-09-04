#pragma once

#include <getopt.h>
#include <string>

//////////////////////////////
//////////////////////////////

#define DEBUG false
#define USE_CSC false
#define OUTPUT_RESULT ""
// Paths are relative to the FPGA implementations!
#define DEFAULT_CSC_FOLDER "data/graphs/other/test"
#define DEFAULT_MTX_FILE "../../data/graph_small_c16.mtx"
#define XCLBIN "../pagerank.xclbin"

// PageRank-specific options;
#define MAX_ITER 100
#define ALPHA 0.8
#define MAX_ERROR 0.000001

//////////////////////////////
//////////////////////////////

struct Options {

    // Input-specific options;
    bool use_csc = USE_CSC;
    std::string graph_path = DEFAULT_MTX_FILE;
    bool use_sample_graph = false;
    bool undirect_graph = false;

    // Testing options;
    uint num_tests = 1;
    int debug = DEBUG;
    std::string output_result = "";

    // FPGA-specific options;
    std::string xclbin_path = XCLBIN;

    // PageRank-specific options;
    int max_iter = MAX_ITER;
    float alpha = ALPHA;
    float max_err = MAX_ERROR;

    //////////////////////////////
    //////////////////////////////

    Options(int argc, char *argv[]) {
        // g: path to the directory that stores the input graph, stored as CSC, with pointer, indices, and values files
        // m: maximum number of iterations
        // e: maximum acceptable error
        // a: dangling factor
        // s: use a small example graph instead of the input files
        // d: if present, print all debug information, else a single summary line at the end
        // t: if present, compute PR the specified number of times
        int opt;
        static struct option long_options[] = {{"debug", no_argument, 0, 'd'},
                                               {"use_sample_graph", no_argument, 0, 's'},
                                               {"graph_path", required_argument, 0, 'g'},
                                               {"use_csc", no_argument, 0, 'c'},
                                               {"max_iter", required_argument, 0, 'm'},
                                               {"max_error", required_argument, 0, 'e'},
                                               {"alpha", required_argument, 0, 'a'},
                                               {"num_tests",
                                                required_argument, 0, 't'},
                                               {"undirect", no_argument, 0, 'u'},
                                               {"xclbin", no_argument, 0, 'x'},
                                               {"output_result", required_argument, 0, 'o'},
                                               {0, 0, 0, 0}};
        // getopt_long stores the option index here;
        int option_index = 0;

        while ((opt = getopt_long(argc, argv, "dg:m:e:sca:t:ux:o:", long_options, &option_index)) != EOF) {
            switch (opt) {
                case 'd':
                    debug = true;
                    break;
                case 'g':
                    graph_path = optarg;
                    break;
                case 'm':
                    max_iter = atoi(optarg);
                    break;
                case 'e':
                    max_err = atof(optarg);
                    break;
                case 's':
                    use_sample_graph = true;
                    break;
                case 'c':
                    use_csc = true;
                    break;
                case 'a':
                    alpha = atof(optarg);
                    break;
                case 't':
                    num_tests = atoi(optarg);
                    break;
                case 'u':
                    undirect_graph = true;
                    break;
                case 'x':
                    xclbin_path = optarg;
                    break;
                case 'o':
                    output_result = optarg;
                    break;
                default:
                    break;
            }
        }
    }
};

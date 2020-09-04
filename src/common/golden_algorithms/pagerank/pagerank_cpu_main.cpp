#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include <vector>

#include "../../csc_matrix/csc_matrix.h"
#include "../../utils/utils.hpp"
#include "../../utils/evaluation_utils.hpp"
#include "../../utils/options.hpp"
#include "pagerank_cpu.hpp"

#define DEFAULT_GRAPH "../../../../data/graphs/mtx/graph_small_c16.mtx"

/////////////////////////////
/////////////////////////////

int main(int argc, char *argv[]) {

    /////////////////////////////
    // Input parameters /////////
    /////////////////////////////

    Options options = Options(argc, argv);
    int debug = options.debug;
    bool use_csc = options.use_csc;
    std::string graph_path = options.graph_path;
    int max_iter = options.max_iter;
    float alpha = options.alpha;
    float max_err = options.max_err;
    bool use_sample_graph = options.use_sample_graph;
    uint num_tests = options.num_tests;
    bool undirect_graph = options.undirect_graph;
    std::string output_result = options.output_result;

    /////////////////////////////
    // Load data ////////////////
    /////////////////////////////

    // Use a sample graph if required;
    if (use_sample_graph) {
        if (debug) {
            std::cout << "Using sample graph located at: " << DEFAULT_GRAPH << std::endl;
        }
        use_csc = false;
        graph_path = DEFAULT_GRAPH;
    }

    // Load dataset and create auxiliary vectors
    csc_t input_m;
    if (!use_csc) {
        input_m = load_graph_mtx(graph_path, debug, undirect_graph);
    } else {
        input_m = load_graph_csc(graph_path, debug);
    }

    uint n_edges = input_m.col_val.size();
    uint n_vertices = input_m.col_ptr.size() - 1;

    if (debug) {
        std::cout << "\n----------------\n- Graph Summary -\n----------------" << std::endl;
        std::cout << "- |V| = " << n_vertices << std::endl;
        std::cout << "- |E| = " << n_edges << std::endl;
        std::cout << "- Undirected: " << (undirect_graph ? "True" : "False") << std::endl;
        std::cout << "- Alpha: " << alpha << std::endl;
        std::cout << "- Max. error: " << max_err << std::endl;
        std::cout << "- Max. iterations: " << max_iter << std::endl;
        std::cout << "----------------\n"
                  << std::endl;
    }

    // Setup the PageRank data structure;
    PageRankCPU pr = PageRankCPU(n_vertices, &input_m, max_iter, alpha, max_err);

    /////////////////////////////
    // Execute the kernel ///////
    /////////////////////////////

    std::vector<double> trans_times(num_tests, 0);
    std::vector<double> exec_times(num_tests, 0);
    for (uint i = 0; i < num_tests; i++) {
        if (debug) {
            std::cout << "\nIteration " << i << ")" << std::endl;
        }
        // Reset the result;
        if (i > 0) {
            trans_times[i] = pr.reset(true, debug) / 10e6;
        }

        exec_times[i] = pr.execute(true, debug) / 10e6;

        if (!debug) {
            if (i == 0) {
                std::cout << "fixed_float_width,fixed_float_scale,v,e,"
                          << "execution_time,transfer_time,error_pct,normalized_dcg" << std::endl;
            }
            std::cout << 32 << "," << 32 << "," << n_vertices << ","
                      << n_edges << "," << exec_times[i] << "," << trans_times[i] << ","
                      << 0 << "," << 0 << std::endl;
        }
    }
    if (debug) {
        int old_precision = std::cout.precision();
        std::cout.precision(2);
        std::cout << "\n----------------\n- Results ------\n----------------" << std::endl;
        std::cout << "Mean transfer time:  " << mean(trans_times) << "±" << st_dev(trans_times) << " ms;" << std::endl;
        std::cout << "Mean execution time: " << mean(exec_times) << "±" << st_dev(exec_times) << " ms;" << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout.precision(old_precision);
    }

    // Store the output if required;
    if (output_result.size() > 0) {
        if (debug) {
            std::cout << "Storing PageRank results to " << output_result << std::endl;
        }
        std::ofstream output_file(output_result);
        if (output_file.fail()) {
            std::cerr << "Error opening " << output_result << std::endl;
            std::cerr.flush();
            return EXIT_FAILURE;
        }
        // Store vertex ID and PageRank value;
        for (uint i = 0; i < pr.result.size(); i++) {
            output_file << i << "," << pr.result[i] << std::endl;
        }
        output_file.close();
    }
}

#pragma once

#include <vector>
#include "../../utils/utils.hpp"
#include "../../csc_matrix/csc_matrix.hpp"

#define MAX_ITER 100
#define ALPHA 0.8
#define MAX_ERROR 0.000001

struct PageRankCPU {
    PageRankCPU(int N, csc_t *input_graph,
                uint max_iter = MAX_ITER, float alpha = ALPHA, float max_error = MAX_ERROR) : N(N), input_graph(input_graph), max_iter(max_iter), alpha(alpha), max_error(max_error) {

        // Initialize support arrays;
        E = input_graph->col_idx.size();
        result = std::vector<double>(N, 1.0 / N);
        result_temp = std::vector<double>(N, 1.0 / N);
        outdegrees = std::vector<unsigned int>(N, 0);
        // Compute the outdegrees;
        for (int v : input_graph->col_idx) {
            outdegrees[v]++;
        }
        for (int i = 0; i < N; i++) {
            if (!outdegrees[i]) {
                dangling_bitmap.push_back(i);
            }
        }
    }

    double execute(bool measure_time = true, bool debug = false);
    double reset(bool measure_time = true, bool debug = false);

    // Instance parameters;
    const int N;
    int E;
    csc_t *input_graph;
    int max_iter;
    float alpha;
    float max_error;

    // Support arrays;
    std::vector<double> result;
    std::vector<double> result_temp;
    std::vector<unsigned int> dangling_bitmap;
    std::vector<unsigned int> outdegrees;
};

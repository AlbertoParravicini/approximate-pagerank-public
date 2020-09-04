#include <chrono>
#include "pagerank_cpu.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

double PageRankCPU::reset(bool measure_time, bool debug) {
    if (debug) {
        std::cout << "Reset values" << std::endl;
    }

    auto start = clock_type::now();

    // Reset the PageRank vectors;
    std::fill(result_temp.begin(), result_temp.end(), 1.0 / N);
    std::fill(result.begin(), result.end(), 1.0 / N);
    auto end = clock_type::now();
    if (measure_time) {
        return (double)chrono::duration_cast<chrono::milliseconds>(end - start).count();
    } else {
        return 0;
    }
}

double PageRankCPU::execute(bool measure_time, bool debug) {

    auto start = clock_type::now();

    uint curr_iter = 0;
    while (curr_iter < max_iter) {

        double dangling_factor = 0;
        for (int v : dangling_bitmap) {
            dangling_factor += result_temp[v];
        }
        dangling_factor *= alpha / N;

        for (int v = 0; v < N; v++) {
            double sum = 0;
            for (int i = input_graph->col_ptr[v]; i < input_graph->col_ptr[v + 1]; i++) {
                sum += result_temp[input_graph->col_idx[i]] / outdegrees[input_graph->col_idx[i]];
            }
            result[v] = (1.0 - alpha) / N + alpha * sum + dangling_factor;
        }
        // Compute L1 norm;
        double l1_norm = 0;
        for (uint i = 0; i < result.size(); i++) {
            l1_norm += std::abs(result[i] - result_temp[i]);
        }
        result_temp = result;
        if (l1_norm <= max_error) {
            break;
        }
        curr_iter++;
    }

    if (debug) {
        std::cout << "Computation terminated in " << curr_iter << " iterations, result:" << std::endl;
		print_array_indexed(result);
    }

    auto end = clock_type::now();
    if (measure_time) {
        return (double)chrono::duration_cast<chrono::milliseconds>(end - start).count();
    } else {
        return 0;
    }
}

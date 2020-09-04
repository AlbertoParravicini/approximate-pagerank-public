//
// Created by Francesco Sgherzi on 8/10/19.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>  /* For O_RDWR, O_WRONLY */
#include <unistd.h> /* For open(), read(), ... */
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <ap_int.h>
#include <chrono>

#include "opencl_utils.hpp"
#include "../../common/csc_matrix/csc_matrix.hpp"
#include "../../common/utils/options.hpp"
#include "../../common/utils/utils.hpp"
#include "../../common/utils/evaluation_utils.hpp"
#include "pagerank_csc.hpp"
#include "pagerank_coo.hpp"
#include "gold_algorithms.hpp"
#include "csc_fpga/csc_fpga.hpp"
#include "coo_fpga/coo_fpga.hpp"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

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
//	std::string graph_path = "../../data/graphs/mtx/graph_small_c16.mtx";
	int max_iter = options.max_iter;
	fixed_float alpha = options.alpha;
	fixed_float max_err = options.max_err;
	bool use_sample_graph = options.use_sample_graph;
	uint num_tests = options.num_tests;
	bool undirect_graph = options.undirect_graph;
	std::string xclbin_path = options.xclbin_path;

	std::string golden_res = "../../data/graphs/other/test/results.txt";

	/////////////////////////////
	// Setup ////////////////////
	/////////////////////////////

	csv_results_t results;
	results.fixed_float_width = FIXED_WIDTH;
	results.fixed_float_scale = FIXED_INTEGER_PART;

	std::vector<std::string> target_devices = { "xilinx_u200_xdma_201830_2" };
	// When running software emulation the program is launched from Emulation-SW/approximate_pagerank-Default;
	std::vector<std::string> kernels = { xclbin_path };
	std::string kernel_name = "multi_ppr_main";

	ConfigOpenCL config(kernel_name);
	setup_opencl(config, target_devices, kernels, debug);

	/////////////////////////////
	// Load data ////////////////
	/////////////////////////////

	// Use a sample graph if required;
	if (use_sample_graph) {
		if (debug) {
			std::cout << "Using sample graph located at: " << DEFAULT_MTX_FILE
					<< std::endl;
		}
		use_csc = false;
		graph_path = DEFAULT_MTX_FILE;
	}

	// Load dataset and create auxiliary vectors;
	auto start_1 = clock_type::now();
	csc_t input_m;
	if (!use_csc) {
		input_m = load_graph_mtx(graph_path, debug, undirect_graph);
	} else {
		input_m = load_graph_csc(graph_path, debug);
	}
	auto end_1 = clock_type::now();
	auto start_2 = clock_type::now();
	csc_fixed_fpga_t f_input_m = convert_to_fixed_point_fpga(input_m);
	auto end_2 = clock_type::now();
	// Convert CSC to COO;
	auto start_3 = clock_type::now();
	coo_fixed_fpga_t coo(f_input_m, true);
	auto end_3 = clock_type::now();

	auto loading_time = chrono::duration_cast < chrono::milliseconds
			> (end_1 - start_1).count();
	auto to_fixed_time = chrono::duration_cast < chrono::milliseconds
			> (end_2 - start_2).count();
	auto to_coo = chrono::duration_cast < chrono::milliseconds
			> (end_3 - start_3).count();

	results.n_edges = f_input_m.col_val.size();
	results.n_vertices = f_input_m.col_ptr.size() - 1;
	std::vector<index_type, aligned_allocator<index_type>> personalization_vertices(
	N_PPR_VERTICES, 0);
	for (int i = 0; i < N_PPR_VERTICES; ++i) {
		personalization_vertices[i] = rand() % results.n_vertices;
	}

	if (debug) {
		std::cout << "\n----------------\n- Graph Summary -\n----------------"
				<< std::endl;
		std::cout << "- |V| = " << results.n_vertices << std::endl;
		std::cout << "- |E| = " << results.n_edges << std::endl;
		std::cout << "- Undirected: " << (undirect_graph ? "True" : "False")
				<< std::endl;
		std::cout << "- Alpha: " << alpha << std::endl;
		std::cout << "- Max. error: " << max_err << std::endl;
		std::cout << "- Max. iterations: " << max_iter << std::endl;
		std::cout << "- Personalization vertices: "
				<< format_array(personalization_vertices.data(), N_PPR_VERTICES)
				<< std::endl;
		std::cout << "----------------" << std::endl;
		std::cout << "- Loading time: " << loading_time / 1000 << " sec"
				<< std::endl;
		std::cout << "- Float-to-fixed time: " << to_fixed_time / 1000 << " sec"
				<< std::endl;
		std::cout << "- CSC-to-COO: " << to_coo / 1000 << " sec" << std::endl;
		std::cout << "- Tot. time: "
				<< (loading_time + to_fixed_time + to_coo) / 1000 << " sec"
				<< std::endl;
		if (!use_csc) {
			long filesize_mb = get_file_size(graph_path) / 1000000;
			double throughput =
					loading_time > 0 ? 1000 * filesize_mb / loading_time : 0;
			std::cout << "- File size: " << filesize_mb << " MB" << std::endl;
			std::cout << "- Throughput: " << throughput << " MB/sec"
					<< std::endl;
		}
		std::cout << "----------------" << std::endl;
	}

	// Setup the PageRank data structure;
	PageRankCOO pr = PageRankCOO(results.n_vertices, results.n_edges, &coo,
			max_iter, alpha, max_err, personalization_vertices);

	/////////////////////////////
	// Setup kernel /////////////
	/////////////////////////////

	// Create Kernel Arguments
	pr.preprocess_inputs();
	pr.setup_inputs(config, debug);
	results.transfer_time = pr.transfer_input_data(config, true, debug);

	/////////////////////////////
	// Golden PageRank //////////
	/////////////////////////////
	auto pr_vec_golden = std::vector<float>(pr.N, 0);
	auto tmp_pr_golden = std::vector<float>(pr.N, 0);
	unsigned int result_iterations_to_convergence = 0;
	unsigned int max_iter_local = max_iter;
	unsigned int local_N = pr.N;
	unsigned int local_E = pr.E;
	auto start_golden = clock_type::now();
	auto results_golden = multi_personalized_pagerank_golden(
			input_m.col_ptr.data(), input_m.col_idx.data(),
			input_m.col_val.data(), &local_N, &local_E, pr_vec_golden.data(),
			pr.dangling_bitmap.data(), tmp_pr_golden.data(), &options.max_err,
			&options.alpha, &max_iter_local, &result_iterations_to_convergence,
			personalization_vertices.data(), N_PPR_VERTICES);
	auto end_golden = clock_type::now();
	if (debug) {
		std::cout << "Golden PageRank converged in " << chrono::duration_cast
				< chrono::milliseconds
				> (end_golden - start_golden).count() << " ms and "
						<< result_iterations_to_convergence << " iterations"
						<< std::endl;

		std::cout << "PR CPU" << std::endl;
		print_matrix_indexed(results_golden, 20, N_PPR_VERTICES);
	}

	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	std::vector<double> trans_times(num_tests, 0);
	std::vector<double> exec_times(num_tests, 0);
	std::vector<std::string> errors(num_tests);
	std::vector<std::string> ndcgs(num_tests);
	std::vector<std::string> edit_distances(num_tests);
	std::vector<double> all_errors;
	std::vector<double> all_ndcgs;
	for (uint i = 0; i < num_tests; i++) {
		if (debug) {
			std::cout << "\nIteration " << i << ")" << std::endl;
		}
		// Reset the result;
		if (i > 0) {
			results.transfer_time = pr.reset(config, true, debug);
		}
		trans_times[i] = results.transfer_time;
		results.execution_time = pr.execute(config, true, debug);

		exec_times[i] = results.execution_time;

		auto results_fixed_idx = std::vector<std::vector<int>>();
		auto results_golden_idx = std::vector<std::vector<int>>();
		for (int j = 0; j < N_PPR_VERTICES; ++j) {
			int cur_begin = j * pr.N;
			int cur_end = (j + 1) * pr.N;
			auto tmp = std::vector<fixed_float>(pr.result.data() + cur_begin,
					pr.result.data() + cur_end);
			auto tmp_sorted = sort_pr(pr.N, tmp.data());
			results_fixed_idx.push_back(tmp_sorted);

			results_golden_idx.push_back(
					sort_pr(pr.N, results_golden[j].data()));
		}

		std::vector<unsigned int> cur_errs(N_PPR_VERTICES, 0);
		for (unsigned int j = 0; j < pr.N; ++j) {
			for (int k = 0; k < N_PPR_VERTICES; ++k) {
				if (results_golden_idx[k][j] != results_fixed_idx[k][j]) {
					cur_errs[k] += 1;
				}
			}
		}

		if (debug) {
			std::cout << "Number of errors -> "
					<< format_array(cur_errs.data(), N_PPR_VERTICES)
					<< std::endl;
		}
		std::vector<std::vector<double>> temp_ndcg(N_PPR_VERTICES);
		std::vector<std::vector<int>> temp_errors(N_PPR_VERTICES);
		std::vector<std::vector<unsigned int>> temp_edit_dist(N_PPR_VERTICES);
		auto bounds = std::vector<int> { 10, 20, 50 };

		for (int j = 0; j < N_PPR_VERTICES; ++j) {
			if (debug)
				std::cout << "Personalization vector " << j << ")" << std::endl;
			temp_ndcg[j] = bounded_ndcg(results_golden_idx[j],
					results_fixed_idx[j], bounds, debug);
			temp_errors[j] = bounded_count_errors(results_golden_idx[j],
					results_fixed_idx[j], bounds, debug);
			temp_edit_dist[j] = bounded_edit_distance(results_golden_idx[j],
					results_fixed_idx[j], bounds, debug);

		}

		// Store error count and NDCGS for each personalization vertex inside a ";" separated list;
		// Top 10, 20 and 50 values are | ("pipe") separated
		std::string curr_err_str = "";
		std::string curr_ndcgs_str = "";
		std::string curr_edit_str = "";
		for (int j = 0; j < N_PPR_VERTICES; ++j) {

			std::string tmp_ndcgs_str = "";
			std::string tmp_err_str = "";
			std::string tmp_edit_str = "";
			for (int k = 0; k < bounds.size(); ++k) {
				tmp_ndcgs_str += std::to_string(temp_ndcg[j][k]) + "|";
				tmp_err_str += std::to_string(temp_errors[j][k]) + "|";
				tmp_edit_str += std::to_string(temp_edit_dist[j][k]) + "|";
			}

			curr_ndcgs_str += tmp_ndcgs_str + ";";
			curr_err_str += tmp_err_str + ";";
			curr_edit_str += tmp_edit_str + ";";

			all_errors.push_back((double) cur_errs[j] / pr.N);
			all_ndcgs.push_back(mean(temp_ndcg[j]));
		}
		errors[i] = curr_err_str;
		ndcgs[i] = curr_ndcgs_str;
		edit_distances[i] = curr_edit_str;

		if (!debug) {
			if (i == 0) {
				std::cout << "fixed_float_width,fixed_float_scale,v,e,"
						<< "execution_time,transfer_time,error_pct,normalized_dcg, edit_dist"
						<< std::endl;
			}
			std::cout << results.fixed_float_width << ","
					<< results.fixed_float_scale << "," << results.n_vertices
					<< "," << results.n_edges << "," << results.execution_time
					<< "," << results.transfer_time << "," << curr_err_str
					<< "," << curr_ndcgs_str << "," << curr_edit_str
					<< std::endl;
		}
	}
	if (debug) {
		int old_precision = cout.precision();
		cout.precision(2);
		std::cout << "\n----------------\n- Results ------\n----------------"
				<< std::endl;
		std::cout << "Mean transfer time:  " << mean(trans_times) << "±"
				<< st_dev(trans_times) << " ms;" << std::endl;
		std::cout << "Mean execution time: " << mean(exec_times) << "±"
				<< st_dev(exec_times) << " ms;" << std::endl;
		std::cout << "Mean % error: " << mean(all_errors) << "±"
				<< st_dev(all_errors) << " %;" << std::endl;
		std::cout << "Mean NDCG: " << mean(all_ndcgs) << "±"
				<< st_dev(all_ndcgs) << ";" << std::endl;
		std::cout << "----------------" << std::endl;
		cout.precision(old_precision);
	}
}


#pragma once

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <set>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include "../ip/csc_matrix.h"


template<typename T>
inline std::vector<int> sort_pr(size_t DIM, T *pr) {

	std::map<int, T> pr_map;
	std::vector<std::pair<int, T>> sorted_pr;
	std::vector<int> sorted_pr_idxs;

	for (uint i = 0; i < DIM; ++i) {
		sorted_pr.push_back( { i, pr[i] });
		pr_map[i] = pr[i];
	}

	std::sort(sorted_pr.begin(), sorted_pr.end(),
			[](const std::pair<int, num_type> &l, const std::pair<int, num_type> &r) {
				if (l.second != r.second)return l.second > r.second;
				else return l.first > r.first;
			});

	for (auto const &pair : sorted_pr) {
		sorted_pr_idxs.push_back(pair.first);
	}
	return sorted_pr_idxs;
}

template<typename T>
inline int compare_results(const size_t DIM, T *pr, std::string golden_result_path, bool debug = false) {

	auto sorted_pr_idxs = sort_pr(DIM, pr);

	if (debug) {
		std::cout << "Checking results..." << std::endl;
	}
	std::ifstream results;
	results.open(golden_result_path);

	int i = 0;
	int tmp = 0;
	int errors = 0;

	int prev_left_idx = 0;
	int prev_right_idx = 0;

	while (results >> tmp) {
		if (debug) {
			std::cout << "Comparing " << tmp << " ==? " << sorted_pr_idxs[i] << std::endl;
		}

		if (tmp != sorted_pr_idxs[i]) {
			if (prev_left_idx != sorted_pr_idxs[i] || prev_right_idx != tmp) {
				errors++;
			}

			prev_left_idx = tmp;
			prev_right_idx = sorted_pr_idxs[i];

		}
		i++;
	}

	if (debug) {
		std::cout << "Percentage of error: " << (((double) errors) / (DIM)) * 100 << "%\n" << std::endl;

		std::cout << "End of computation! Freeing memory..." << std::endl;
	}

	return errors;

}

template<typename T>
inline double normalized_discounted_cumulative_gain(const size_t DIM, T *vec, std::string golden_results_path,
		const bool debug = false) {

	std::ifstream results_stream;
	std::vector<int> results;
	std::vector<int> pr = sort_pr(DIM, vec);
	std::unordered_map<int, int> ranking;
	std::unordered_map<int, int> golden_ranking;
	int tmp;
	double dcg = 0.0;
	double idcg = 0.0;

	results_stream.open(golden_results_path);

	while (results_stream >> tmp) {
		results.push_back(tmp);
	}

	for (uint i = 0; i < DIM; ++i) {
		ranking[pr[i]] = DIM - i;
		golden_ranking[results[i]] = DIM - i;
	}

	for (uint i = 0; i < DIM; ++i) {
		int elem = ranking[i];
		int golden_elem = golden_ranking[i];

		if (debug) {
			std::cout << "elem: " << elem << " golden: " << golden_elem << " i -> " << i << std::endl;
		}

		dcg += (double) elem / log2(i + 2);
		idcg += (double) golden_elem / log2(i + 2);

	}

	return dcg / idcg;
}

/////////////////////////////
/////////////////////////////

template<typename T>
inline T mean(std::vector<T> x) {
	T sum = 0;
	for (uint i = 0; i < x.size(); i++) {
		sum += x[i];
	}
	return sum / x.size();
}

template<typename T>
inline T st_dev(std::vector<T> x) {
	T mean = 0;
	T mean_sq = 0;
	for (uint i = 0; i < x.size(); i++) {
		mean += x[i];
		mean_sq += x[i] * x[i];
	}
	T diff = mean_sq - mean * mean / x.size();
	diff = diff >= 0 ? diff : 0;
	return std::sqrt(diff / x.size());
}

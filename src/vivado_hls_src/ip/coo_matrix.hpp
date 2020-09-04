//
// Created by fra on 19/04/19.
//

#pragma once

#include <iostream>
#include <vector>
#include <limits.h>
#include "csc_matrix.hpp"

struct coo_t {
    std::vector<index_type> start;
    std::vector<index_type> end;
    std::vector<num_type> val;

    coo_t(std::vector<index_type> _start, std::vector<index_type> _end, std::vector<num_type> _val): start(_start), end(_end), val(_val) {}

    // Create a COO from a CSC. Note that the COO will contain the graph transposed, i.e. edges are ingoing;
    coo_t(csc_t csc) {
		for (int i = 0; i < csc.col_ptr.size(); i++) {
    		int s = csc.col_ptr[i];
    		int e = csc.col_ptr[i + 1];
    		for (int j = s; j < e; j++) {
    			start.push_back(i);
    			end.push_back(csc.col_idx[j]);
    			val.push_back(csc.col_val[i]);
    		}
    	}
    }

    void print_coo(bool compact = false, bool transposed = true) {
    	if (compact) {
    		// TODO: need to add transposed;
    		index_type last_s = 0;
    		index_type curr_e = 0;
    		std::vector<index_type> neighbors;
    		std::vector<num_type> vals;
    		for (int i = 0; i < start.size(); i++) {
    			index_type curr_s = start[i];
    			if (curr_s == last_s) {
    				neighbors.push_back(end[curr_e++]);
    				vals.push_back(val[curr_e++]);
    			} else {
    				neighbors = { end[curr_e++] };
    				vals = { val[curr_e++] };
    				std::cout << i << ") degree: " << neighbors.size() << std::endl;
    				std::cout << "edges: ";
    				for (auto e: neighbors) {
    					std::cout << e << ", ";
    				}
    				std::cout << std::endl;
    				std::cout << "val: ";
    				for (auto v: vals) {
						std::cout << v << ", ";
					}
					std::cout << std::endl;
    			}
    		}
    	} else {
        	for (int i = 0; i < start.size(); i++) {
        		std::cout << start[i] << (transposed ? " <- " : " -> ") << end[i] << ": " << val[i] << std::endl;
        	}
    	}
    }
};

struct coo_fixed_t {
	std::vector<index_type> start;
	    std::vector<index_type> end;
	    std::vector<fixed_num_type> val;
};

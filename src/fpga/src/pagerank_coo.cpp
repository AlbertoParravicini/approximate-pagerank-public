#include "pagerank_coo.hpp"

/////////////////////////////
/////////////////////////////

void PageRankCOO::setup_inputs(ConfigOpenCL &config, bool debug) {

	if (debug) {
		std::cout << "Create Kernel Arguments" << std::endl;
	}

	d_coo_start = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_E, start_in.data());
	d_coo_end = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_E, end_in.data());
	d_coo_val = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_E, val_in.data());
	d_result = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_N * N_PPR_VERTICES, result_out.data());
	d_dangling_bitmap = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_bitmap, dangling_bitmap_in.data());
	d_personalization_vertices = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(index_type) * N_PPR_VERTICES, personalization_vertices.data());

	int narg = 0;
	config.kernel.setArg(narg++, d_coo_start);
	config.kernel.setArg(narg++, d_coo_end);
	config.kernel.setArg(narg++, d_coo_val);
	config.kernel.setArg(narg++, N);
	config.kernel.setArg(narg++, E_fixed);
	config.kernel.setArg(narg++, d_result);
	config.kernel.setArg(narg++, d_dangling_bitmap);
	config.kernel.setArg(narg++, max_error);
	config.kernel.setArg(narg++, alpha);
	config.kernel.setArg(narg++, max_iter);
	config.kernel.setArg(narg++, d_personalization_vertices);
}

/////////////////////////////
/////////////////////////////

void PageRankCOO::preprocess_inputs() {

	// Change type to dangling_bitmap
	std::vector<dangling_type> dangling_bitmap_fpga;
	for (unsigned int &el : dangling_bitmap) {
		dangling_bitmap_fpga.push_back(el);
	}

	// Initialize all vectors of length N
	result_out = std::vector<input_block, allocator>(num_blocks_N * N_PPR_VERTICES);

	// Initialize all vectors of length E
	start_in = std::vector<input_block, allocator>(num_blocks_E);
	end_in = std::vector<input_block, allocator>(num_blocks_E);
	val_in = std::vector<input_block, allocator>(num_blocks_E);

	dangling_bitmap_in = std::vector<input_block, allocator>(num_blocks_bitmap);

	// Pack vector in chunks of 512 bits each;

	write_packed_array(input_graph->start.data(), start_in.data(), E_fixed, num_blocks_E);
	write_packed_array(input_graph->end.data(), end_in.data(), E_fixed, num_blocks_E);
	write_packed_array(input_graph->val.data(), val_in.data(), E_fixed, num_blocks_E);

	write_packed_matrix(result.data(), result_out.data(), N_PPR_VERTICES, N, num_blocks_N);
	write_packed_array(dangling_bitmap_fpga.data(), dangling_bitmap_in.data(), N, num_blocks_bitmap, AP_UINT_BITWIDTH, 1);
}

/////////////////////////////
/////////////////////////////

double PageRankCOO::transfer_input_data(ConfigOpenCL &config, bool measure_time, bool debug) {

	if (debug) {
		std::cout << "Write inputs into device memory" << std::endl;
	}

	// Transfer data from host to device (0 means host-to-device transfer);
	cl::Event e;
	config.queue.enqueueMigrateMemObjects( { d_coo_start, d_coo_end, d_coo_val, d_dangling_bitmap, d_personalization_vertices }, 0, NULL, &e);
	e.wait();

	if (measure_time) {
		double elapsed = get_event_execution_time(e);
		if (debug) {
			std::cout << "Data transfer took " << elapsed / 10e6 << " ms" << std::endl;
		}
		return elapsed / 10e6;
	} else {
		return -1;
	}
}

/////////////////////////////
/////////////////////////////

double PageRankCOO::execute(ConfigOpenCL &config, bool measure_time, bool debug) {


	if (debug) {
		std::cout << "Execute the kernel" << std::endl;
	}
	auto start = clock_type::now();
	config.queue.enqueueTask(config.kernel);
	config.queue.finish();

	// Read back the results from the device to verify the output
	config.queue.enqueueMigrateMemObjects( { d_result }, CL_MIGRATE_MEM_OBJECT_HOST);
	config.queue.finish();
	auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();

	// Unpack the result vector;
	read_packed_matrix(result.data(), N_PPR_VERTICES, N, result_out.data(), num_blocks_N);

	if (debug) {
		std::cout << "Kernel terminated" << std::endl;
		print_matrix_indexed(result.data(), N_PPR_VERTICES, N, N_PPR_VERTICES, 20);
	}
	if (measure_time) {
		if (debug) {
			std::cout << "Computation took " << elapsed / 1e6 << " ms" << std::endl;
		}
		return elapsed / 1e6;
	} else {
		return -1;
	}
}

/////////////////////////////
/////////////////////////////

double PageRankCOO::reset(ConfigOpenCL &config, bool measure_time, bool debug) {
	if (debug) {
		std::cout << "Reset values not necessary" << std::endl;
	}

	if (measure_time) {
		return 0;
	} else {
		return -1;
	}
}

////////////////////////////
////////////////////////////

void PageRankCOO::initialize_dangling_bitmap() {
	for (uint i = 0; i < input_graph->end.size(); i++) {
		dangling_bitmap[input_graph->end[i]] = 0;
	}
}

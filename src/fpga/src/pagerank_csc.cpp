#include "pagerank_csc.hpp"

/////////////////////////////
/////////////////////////////

void PageRankCSC::setup_inputs(ConfigOpenCL &config, bool debug) {

	if (debug) {
		std::cout << "Create Kernel Arguments" << std::endl;
	}

	d_csc_col_ptr = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_N, ptr_in.data());
	d_csc_col_idx = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_E, idx_in.data());
	d_csc_col_val = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_E, val_in.data());
	d_result = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_N, result_out.data());
	d_pr = cl::Buffer(config.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_N, pr_in.data());
	d_dangling_bitmap = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_bitmap, dangling_bitmap_in.data());
	d_pr_tmp = cl::Buffer(config.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(input_block) * num_blocks_N, pr_tmp_in.data());

	int narg = 0;
	config.kernel.setArg(narg++, d_csc_col_ptr);
	config.kernel.setArg(narg++, d_csc_col_idx);
	config.kernel.setArg(narg++, d_csc_col_val);
	config.kernel.setArg(narg++, N);
	config.kernel.setArg(narg++, E);
	config.kernel.setArg(narg++, d_result);
	config.kernel.setArg(narg++, d_pr);
	config.kernel.setArg(narg++, d_dangling_bitmap);
	config.kernel.setArg(narg++, d_pr_tmp);
	config.kernel.setArg(narg++, max_error);
	config.kernel.setArg(narg++, alpha);
	config.kernel.setArg(narg++, max_iter);
}

/////////////////////////////
/////////////////////////////

void PageRankCSC::preprocess_inputs() {
	auto val = input_graph->col_val;
	auto ptr = input_graph->col_ptr;
	auto idx = input_graph->col_idx;

	// Change type to dangling_bitmap
	std::vector<dangling_type> dangling_bitmap_fpga;
	for (unsigned int &el : dangling_bitmap) {
		dangling_bitmap_fpga.push_back(el);
	}

	// Initialize all vectors of length N
	ptr_in = std::vector<input_block, allocator>(num_blocks_N);
	pr_in = std::vector<input_block, allocator>(num_blocks_N);
	pr_tmp_in = std::vector<input_block, allocator>(num_blocks_N);
	result_out = std::vector<input_block, allocator>(num_blocks_N);

	// Initialize all vectors of length E
	idx_in = std::vector<input_block, allocator>(num_blocks_E);
	val_in = std::vector<input_block, allocator>(num_blocks_E);

	dangling_bitmap_in = std::vector<input_block, allocator>(num_blocks_bitmap);

	// Pack vector in chunks of 512 bits each;

	// Remove the starting 0 from the ptr vector, to align it with other vectors.
	// Hence, read with a + 1 index;
	write_packed_array(ptr.data() + 1, ptr_in.data(), N, num_blocks_N);
	write_packed_array(pr.data(), pr_in.data(), N, num_blocks_N);
	write_packed_array(pr_tmp.data(), pr_tmp_in.data(), N, num_blocks_N);
	write_packed_array(result.data(), result_out.data(), N, num_blocks_N);
	write_packed_array(dangling_bitmap_fpga.data(), dangling_bitmap_in.data(), N, num_blocks_bitmap, AP_UINT_BITWIDTH, 1);
	write_packed_array(idx.data(), idx_in.data(), E, num_blocks_E);
	write_packed_array(val.data(), val_in.data(), E, num_blocks_E);
}

/////////////////////////////
/////////////////////////////

double PageRankCSC::transfer_input_data(ConfigOpenCL &config, bool measure_time, bool debug) {

	if (debug) {
		std::cout << "Write inputs into device memory" << std::endl;
	}

	// Transfer data from host to device (0 means host-to-device transfer);
	cl::Event e;
	config.queue.enqueueMigrateMemObjects( { d_csc_col_ptr, d_csc_col_idx, d_csc_col_val, d_pr, d_dangling_bitmap, d_pr_tmp }, 0, NULL, &e);
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

double PageRankCSC::execute(ConfigOpenCL &config, bool measure_time, bool debug) {


	if (debug) {
		std::cout << "Execute the kernel" << std::endl;
	}
	auto start = clock_type::now();
	config.queue.enqueueTask(config.kernel);
	config.queue.finish();

	// Read back the results from the device to verify the output
	config.queue.enqueueMigrateMemObjects( { d_result }, CL_MIGRATE_MEM_OBJECT_HOST);
	config.queue.finish();
	auto elapsed = chrono::duration_cast<chrono::milliseconds>(clock_type::now() - start).count();

	// Unpack the result vector;
	read_packed_array(result.data(), result_out.data(), N, num_blocks_N);

	if (debug) {
		std::cout << "Kernel terminated" << std::endl;
		print_array_indexed(result);
	}
	if (measure_time) {
		if (debug) {
			std::cout << "Computation took " << elapsed << " ms" << std::endl;
		}
		return elapsed;
	} else {
		return -1;
	}
}

/////////////////////////////
/////////////////////////////

double PageRankCSC::reset(ConfigOpenCL &config, bool measure_time, bool debug) {
	if (debug) {
		std::cout << "Reset values" << std::endl;
	}

	// Reset the pr_in vector;
	std::fill(pr.begin(), pr.end(), 1.0 / N);
	// Reset the result vector (not necessary, however);
	std::fill(result.begin(), result.end(), 1.0 / N);
	// Re-pack the array;
	write_packed_array(pr.data(), pr_in.data(), N, num_blocks_N);

	cl::Event e;
	config.queue.enqueueMigrateMemObjects( { d_pr }, 0, NULL, &e);
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

////////////////////////////
////////////////////////////

void PageRankCSC::initialize_dangling_bitmap() {
	for (uint i = 0; i < input_graph->col_idx.size(); ++i) {
		dangling_bitmap[input_graph->col_idx[i]] = 0;
	}
}

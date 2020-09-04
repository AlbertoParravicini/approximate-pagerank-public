Alberto Parravicini, Francesco Sgherzi

Board used: Alveo U200, FPGA: xcu200-fsgd2104-2-e

Software Version: Xilinx Vivado HLS 2018.3, Xilinx SDx 2018.3

Description of archive:

* repository: it contains a copy of the source files used for the project, and the source code of the server used in the demo. The "data" subfolder contains a couple of small graphs that can be used for testing purposes
* demo-frontend: source code of the front-end of the demo
* bitstream: it contains the bitstream required to execute the project
* report: it contains the report for the project

Instructions to build and test project:

* The file `repository/src/fpga/src/csc_fpga/csc_fpga.hpp` contains different useful settings. Default values will be ok to test the project, but you can play with them if desired
	* FIXED_WIDTH: bit-width of fixed-point values, including 1 bit of integer part
	* N_PPR_VERTICES: number of vertices for which PPR is computed in parallel
	* AP_UINT_BITWIDTH: how many COO values (i.e. graph edges) are processed for each clock cycle
	* MAX_VERTICES: maximum number of vertices supported by the graph. The default value (2^14) is small to allow faster build time, and is enough to use the small graphs included in the repository
* To build the project from scratch, and run it on FPGA, perform the following steps
	* `cd repository`
	* `make build TARGET=hw`. This will start the compilation and bitstream generation. The `TARGET_CLOCK` setting in the Makefile allows to specify the desired clock frequency. Whether this value can be achieved depends on the specified bitwidth and other settings, although SDx will try building the project with the highest clock frequency it can.
	* `make host TARGET=hw`. Optional step, it rebuilds the host code. It might be required if the machine where the `build` step was performed has a different CPU architecture than the machine where the compiled host code is executed. If you get errors such as `HAL version 1 not supported` when executing the code, you need to rebuild the host code
	* The file `repository/src/fpga/src/host_demo_cpp` (currently commented out) is used in the demo. It will setup the FPGA and wait indefinitely for data to be written on FIFOs. You can modify the Makefile to compile this file instead, if you want to try the demo by yourself, although the default host file will be enough to test the project and is simpler to build
* The project was tested on an Alveo U200 board with DSA `xilinx_u200_xdma_201830_2`. Remember to `source` the FPGA `setup.sh` on the machine where the code is executed!
* Running the code:
	* `cd repository/build/hw/xilinx_u200_xdma_201830_2`
	* `./approximate_pagerank -d -g <path/to/graph.mtx> -m 100 -t 10 -x <path/to/xclbin>`
	* `-d`: if present, print debug information
	* `-s`: if present, use a small sample graph
	* `-m`: max number of iterations
	* `-e`: error value used as convergence threshold
	* `-a`: value of alpha
	* `-t`: number of times the computation is repeated, for testing
	* `-p`: personalization vertex ID. If not present, use N_PPR_VERTICES random values for testing
	

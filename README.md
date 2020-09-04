# Approximate Pagerank on FPGA

## Guida Intergalattica for Babbi, Version 2

### 1. Setup the FPGA
	
* On Gozzo:  

* Setup the FPGA: `source /home/user/coursera_dsa/xbinst/setup.sh`  
* Check that it works: 

`cd /home/user/Xilinx/SDx/2018.2/7v3_dsa/xbinst/test`
`./verify.exe`
	
* Check with `sdxsyschk` the platform name.
* Or use `sdxsyschk | grep <platform_short_name (e.g. 7v3)>`

### 2. Recompile the host

* When running an OpenCL host build on NAGS30, you might get errors like `HAL version 1 not supported`.
* If so, recompile the host on the current machine.
* To recompile the host:  

	`make host_gozzo`

* The executable is inside `build/fpga`
* Make sure the bitstream is located in the `build` folder (if not, specify its path when running the executable).

### 3. Run tests

* To run tests on CPU/FPGA/GPU, `cd src/resources/python/run`.
* Then, run the script `run.sh` or call directly `python3 run.py`.

#######################################################################################################################################
#
#	Basic Makefile for SDx 2018.3
#	Lorenzo Di Tucci, Emanuele Del Sozzo, Alberto Parravicini
#	{lorenzo.ditucci, emanuele.delsozzo, albetro.parravicini}@polimi.it
#	Usage make [emulation | build | clean | clean_sw_emu | clean_hw_emu | clean_hw | cleanall] TARGET=<sw_emu | hw_emu | hw>
#
#
#######################################################################################################################################

XOCC=v++
CC=xcpp

#############################
# Define files to compile ###
#############################

# Host code
HOST_SRC=./src/fpga/src/host.cpp ./src/fpga/src/pagerank_csc.cpp ./src/fpga/src/csc_fpga/csc_fpga.cpp ./src/fpga/src/pagerank_coo.cpp
HOST_HEADER_DIRS=

# Host header files (optional, used to check if rebuild is required);
UTILS_DIR=./src/common/utils
FPGA_DIR=./src/fpga/src
HOST_HEADERS=./src/common/csc_matrix/csc_matrix.hpp ./src/common/coo_matrix/coo_matrix.hpp $(UTILS_DIR)/evaluation_utils.hpp $(UTILS_DIR)/mmio.hpp $(UTILS_DIR)/options.hpp $(UTILS_DIR)/utils.hpp $(FPGA_DIR)/aligned_allocator.h $(FPGA_DIR)/gold_algorithms.hpp $(FPGA_DIR)/opencl_utils.hpp $(FPGA_DIR)/csc_fpga/csc_fpga.hpp $(FPGA_DIR)/coo_fpga/coo_fpga.hpp $(FPGA_DIR)/pagerank_coo.hpp 

# Name of host executable
HOST_EXE=approximate_pagerank

# Kernel
KERNEL_DIR=./src/fpga/src/ip_cores
KERNEL_SRC=$(KERNEL_DIR)/multi_personalized_pagerank.cpp 
KERNEL_HEADER_DIRS=./src/fpga/src/ip_cores
KERNEL_FLAGS=
# Name of the xclbin;
KERNEL_EXE=multi_ppr_main
# Name of the main kernel function to build;
KERNEL_NAME=multi_ppr_main

#############################
# Define FPGA & host flags  #
#############################

# Target clock of the FPGA, in MHz;
TARGET_CLOCK=400
# Port width, in bit, of the kernel;
PORT_WIDTH=256

# Device code for Alveo U200;
ALVEO_U200=xilinx_u200_xdma_201830_2
ALVEO_U200_DEVICE="\"xilinx_u200_xdma_201830_2"\"
TARGET_DEVICE=$(ALVEO_U200)

# Flags to provide to xocc, specify here associations between memory bundles and physical memory banks.
# Documentation: https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/wrj1504034328013.html
KERNEL_LDCLFLAGS=--xp param:compiler.preserveHlsOutput=1 \
	--sp $(KERNEL_NAME)_1.m_axi_gmem0:bank0 \
	--sp $(KERNEL_NAME)_1.m_axi_gmem1:bank1 \
	--sp $(KERNEL_NAME)_1.m_axi_gmem2:bank2 \
	--sp $(KERNEL_NAME)_1.m_axi_gmem3:bank3 \
	--max_memory_ports all \
	--memory_port_data_width $(KERNEL_NAME):$(PORT_WIDTH)

KERNEL_ADDITIONAL_FLAGS=--kernel_frequency $(TARGET_CLOCK) -O3

# Specify host compile flags and linker;
HOST_INCLUDES=-I$(HOST_HEADER_DIRS) -I${XILINX_XRT}/include -I${XILINX_VIVADO}/include
HOST_CFLAGS=$(HOST_INCLUDES) -D TARGET_DEVICE=$(ALVEO_U200_DEVICE) -g -D C_KERNEL -O3 -std=c++14 
HOST_LFLAGS=-L${XILINX_XRT}/lib -lxilinxopencl -lOpenCL

#############################
# Define compilation type ###
#############################

# TARGET for compilation [sw_emu | hw_emu | hw]
TARGET=none
REPORT_FLAG=n
REPORT=
ifeq (${TARGET}, sw_emu)
$(info software emulation)
TARGET=sw_emu
ifeq (${REPORT_FLAG}, y)
$(info creating REPORT for software emulation set to true. This is going to take longer as it will synthesize the kernel)
REPORT=--report estimate
else
$(info I am not creating a REPORT for software emulation, set REPORT_FLAG=y if you want it)
REPORT=
endif
else ifeq (${TARGET}, hw_emu)
$(info hardware emulation)
TARGET=hw_emu
REPORT=--report estimate
else ifeq (${TARGET}, hw)
$(info system build)
TARGET=hw
REPORT=--report system
else
$(info no TARGET selected)
endif

PERIOD:= :
UNDERSCORE:= _
DEST_DIR=build/$(TARGET)/$(subst $(PERIOD),$(UNDERSCORE),$(TARGET_DEVICE))


#############################
# Define targets ############
#############################

clean:
	rm -rf .Xil emconfig.json 

clean_sw_emu: clean
	rm -rf sw_emu
clean_hw_emu: clean
	rm -rf hw_emu
clean_hw: clean
	rm -rf hw

cleanall: clean_sw_emu clean_hw_emu clean_hw
	rm -rf _xocc_* xcl_design_wrapper_*

check_TARGET:
ifeq (${TARGET}, none)
	$(error Target can not be set to none)
endif

host:  check_TARGET $(HOST_SRC) $(HOST_HEADERS)
	mkdir -p $(DEST_DIR)
	$(CC) $(HOST_SRC) $(HOST_CFLAGS) $(HOST_LFLAGS) -o $(DEST_DIR)/$(HOST_EXE)

xo:	check_TARGET
	mkdir -p $(DEST_DIR)
	$(XOCC) --platform $(TARGET_DEVICE) --target $(TARGET) --compile --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(KERNEL_SRC) $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xo

xclbin:  check_TARGET xo
	$(XOCC) --platform $(TARGET_DEVICE) --target $(TARGET) --link --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(DEST_DIR)/$(KERNEL_EXE).xo $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xclbin

emulation:  host xclbin
	export XCL_EMULATION_MODE=$(TARGET)
	emconfigutil --platform $(TARGET_DEVICE) --nd 1
	./$(DEST_DIR)/$(HOST_EXE) $(DEST_DIR)/$(KERNEL_EXE).xclbin
	$(info Remeber to export XCL_EMULATION_MODE=$(TARGET) and run emconfigutil for emulation purposes)

build:  host xclbin

run_system:  build
	./$(DEST_DIR)/$(HOST_EXE) $(DEST_DIR)/$(KERNEL_EXE)


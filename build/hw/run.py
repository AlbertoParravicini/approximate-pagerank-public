#!/bin/python
from os import system

# Used to run tests used for papers.
# Place it in the same folder as the xclbin and the host executable. Change parameters below according to your test;

DATE = "2020_09_03"
DATE2 = "03-09-2020"
GRAPH_PREFIX = "../../../../../approximate-pagerank/data/graphs/mtx/sparsity"
RESULT_PREFIX = f"../../../../../approximate-pagerank/data/results/raw_results/{DATE}"
BITS = 26

for i in [5, 10, 15]:
    for j in range(200 // 8 + 1):
        print("Run " + str(j))
        # Graphs with less than 2^17 vertices;
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_1002178_2020_01_12_10_39_42.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-1002178-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_1000000_2020_01_12_10_39_42.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-1000000-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_999845_2020_01_12_10_39_42.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-999845-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/amazon.mtx > {RESULT_PREFIX}/amazon/coo-stream-{BITS}-8-{i}it-130k-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/twitter.mtx > {RESULT_PREFIX}/twitter/coo-stream-{BITS}-8-{i}it-130k-{DATE2}-{j + 1}.csv 2>/dev/null")

        # Graphs with >= 2^17 vertices, require a compatible bitstream to be executed;
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_200000_1999249_2020_01_23_11_30_35.mtx > {RESULT_PREFIX}/gnp/coo-stream-20-4-{i}it-200000-1999249-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_200000_2000000_2020_01_23_11_30_35.mtx > {RESULT_PREFIX}/smw/coo-stream-20-4-{i}it-200000-2000000-{DATE2}-{j + 1}.csv 2>/dev/null")
        # system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_200000_1999825_2020_01_23_11_30_35.mtx > {RESULT_PREFIX}/pc/coo-stream-20-4-{i}it-200000-1999825-{DATE2}-{j + 1}.csv 2>/dev/null")

        # Scalability benchmarks, less than 2^17 vertices;
        # GNP
	system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_119680_2020_05_16_11_32_37.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-119680-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_499113_2020_05_16_18_34_35.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-499113-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_1001345_2020_05_16_11_15_16.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-1001345-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_5005142_2020_05_16_18_42_38.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-5005142-{DATE2}-{j + 1}.csv 2>/dev/null")

        #system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_10002412_2020_05_16_11_15_35.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-10002412-{DATE2}-{j + 1}.csv 2>/dev/null")
        #system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/gnp_100000_100001132_2020_05_16_11_37_05.mtx > {RESULT_PREFIX}/gnp/coo-stream-{BITS}-8-{i}it-100000-100001132-{DATE2}-{j + 1}.csv 2>/dev/null")

        # SMW
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_100000_2020_05_16_11_42_30.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-100000-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_500000_2020_05_16_18_34_48.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-500000-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_1000000_2020_05_16_11_42_55.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-1000000-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_5000000_2020_05_16_18_44_23.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-5000000-{DATE2}-{j + 1}.csv 2>/dev/null")

        #system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/smw_100000_10000000_2020_05_16_11_43_18.mtx > {RESULT_PREFIX}/smw/coo-stream-{BITS}-8-{i}it-100000-10000000-{DATE2}-{j + 1}.csv 2>/dev/null")

        # PC
	system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_199996_2020_05_16_11_34_42.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-199996-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_499960_2020_05_16_18_35_16.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-499960-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_999849_2020_05_16_11_34_59.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-999849-{DATE2}-{j + 1}.csv 2>/dev/null")
        system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_4995962_2020_05_16_18_36_17.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-4995962-{DATE2}-{j + 1}.csv 2>/dev/null")

        #system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_9985037_2020_05_16_11_35_42.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-9985037-{DATE2}-{j + 1}.csv 2>/dev/null")
        #system(f"./approximate_pagerank -x multi_ppr_main.xclbin -t 1 -m {i} -g {GRAPH_PREFIX}/pc_100000_98683537_2020_05_16_12_11_12.mtx > {RESULT_PREFIX}/pc/coo-stream-{BITS}-8-{i}it-100000-98683537-{DATE2}-{j + 1}.csv 2>/dev/null")





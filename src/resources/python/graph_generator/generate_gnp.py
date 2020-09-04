import networkx as nx
from threading import Thread
from time import time
import os

m = [[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]

def write_log(message, endl='\n'):
    print(message)
    f = open("log/graph_generator-test.log".format(str(time())), "a+")
    if endl == '\n':
        f.write("{}{}".format(message, endl))
    else:
        f.write("{}: {}{} ".format(str(time()), message, endl))

    f.close()


def format_file(filename, values):

    write_log("Formatting {}".format(filename))

    f = open(filename, 'w+')
    for e in values:
        f.write('{}\n'.format(e))

if __name__ == "__main__":

    output_folder = "../../../../data/graphs"

	DIM = 10**5
	write_log("DIM = {}".format(DIM))

	PERC_SPARSE = 0.000003
	write_log("PERC_SPARSE = {}% => expecting {} elements per column".format(str((1 - PERC_SPARSE)*100), DIM * PERC_SPARSE))

	write_log("Generating graph...")
	g = nx.fast_gnp_random_graph(DIM, PERC_SPARSE, directed=True)

	# The following line is used for testing with parra's matrix
	#g = nx.from_numpy_matrix(np.matrix(m), create_using=nx.DiGraph)

	write_log("Computing pagerank... alpha=0.85, max_iter=200, tol=1e-12", endl='')
	start = time()

	pr = nx.pagerank_scipy(g, alpha=0.85, max_iter=200, tol=1e-12)

	write_log("DONE [{}s]".format(time() - start))


	write_log("Sorting and dumping pr values...", endl='')
	start = time()

	pr_sorted = sorted(pr.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

	f = open('generated_csc/test/results.txt', 'w+')

	for el in pr_sorted:
	    f.write(str(el[0]) + '\n')

	f.close()
	write_log("DONE [{}s]".format(time() - start))


	write_log("Formatting matrix in stochastic form...", endl='')
	start = time()

	g = nx.stochastic_graph(g, copy=True)

	write_log("DONE [{}s]".format(time() - start))

	write_log("Converting matrix in csc", endl='')
	start = time()

	csc = nx.to_scipy_sparse_matrix(g, format='csc')


	write_log("DONE [{}s]".format(time() - start))

	data = csc.data
	indices_col = csc.indices
	indptr = csc.indptr

	write_log("Dumping matrices: ")
	start = time()

	data_t = Thread(target=format_file, args=(os.path.join(output_folder, 'col_val.txt').format(PERC_SPARSE), data))
	col_idx_t = Thread(target=format_file, args=(os.path.join(output_folder, 'col_idx.txt').format(PERC_SPARSE), indices_col))
	col_ptr_t = Thread(target=format_file, args=(os.path.join(output_folder, 'col_ptr.txt').format(PERC_SPARSE), indptr))


	data_t.start()
	col_idx_t.start()
	col_ptr_t.start()

	data_t.join()
	col_idx_t.join()
	col_ptr_t.join()

	write_log("DONE [{}s]".format(time() - start))


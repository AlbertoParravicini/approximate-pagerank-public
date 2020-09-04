# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:57:38 2019

@author: albyr
"""

import numpy as np
import time
import networkx as nx
import scipy
from scipy.io import mmread

def timeit(function):
    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)
        te = time.time()
        print(f"{function.__name__}, exec. time: {(te - ts) * 1000:.2f} ms")
        return result
    return timed


@timeit
def pr_1(X, num_iterations=30, alpha=0.85):
    """
    Simple implementation of PageRank inspired by networkx, 
    it computes PageRank through iterative matrix power iterations.

    :input X: adjacency matrix of a graph
    """
    
    N = X.shape[0]
    # Divide each vertex (i.e. column) by its out-degree (i.e. sum of column);
    row_sum = X.sum(1)
    nan_indices = row_sum == 0
    with np.errstate(invalid="ignore", divide="ignore"): # Ignore divide-by-zero warnings;
        row_sum = 1 / row_sum
    row_sum[nan_indices] = np.nan_to_num(row_sum[nan_indices], copy=False) # Fix NaN, i.e. dangling nodes;
    X = np.diag(row_sum).dot(X) 
    
    print(X.T)
    
    # PageRank initialization;
    x = scipy.repeat(1.0 / N, N)
    # Compute a bitmask that indicates dangling vertices;
    dangling_mask = np.array(X.sum(axis=1) == 0, dtype=int)
    
    # Main iteration;
    for i in range(num_iterations):  
        extra = (alpha / N) * np.dot(x, dangling_mask) + (1-alpha) / N
        x = np.dot(x, X)
        x = alpha * x + extra
    
    return x
        

if __name__ == "__main__":
    
#    # Create a random directed graph;
#    num_vertices = 16
#    X = np.array(np.random.randint(0, 2, num_vertices**2), dtype=float).reshape((num_vertices, num_vertices))
#    X += np.identity(num_vertices, dtype=float)
#    X[X > 1] = 1
#    
#    # Or use a fixed matrix instead;
#    X = np.array([
#                 [1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1.],
#                 [0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1.],
#                 [0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0.],
#                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
#                 [0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
#                 [1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0.],
#                 [1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0.],
#                 [1., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.],
#                 [1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1.],
#                 [1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1.],
#                 [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1.],
#                 [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1.],
#                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                 [1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1.],
#                 [0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.]])
    
    # Load a graph from a file;
    # X = np.array(mmread("../../../data/graphs/mtx/graph_small_c16.mtx").todense())
    # X = np.maximum(X - np.eye(X.shape[0]), 0)
    
    X = np.array([[0,0,0,0,0],
              [1,0,0,1,0],
              [1,1,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0]])

    
    personalization_vertex = 1
 
    # PageRank;
    pr_res = pr_1(X.copy(), alpha=0.8, num_iterations=100)       
    print(f"\tsum of pagerank: {sum(pr_res)}")
    
    # Golden values, from networkx;
    G = nx.from_numpy_array(X.copy(), create_using=nx.DiGraph)
    try:
        pr_gold = np.array(list(nx.pagerank(G, personalization={personalization_vertex: 1}, alpha=0.8, max_iter=100, tol=10e-6).values()))
    except nx.exception.PowerIterationFailedConvergence:
        pass
    print("golden pagerank")
    print(f"\tsum of golden pagerank: {sum(pr_gold)}")
    print(f"difference: {sum((pr_res - pr_gold)**2)}")
    print(f"similarity percentage: {100*(1 - sum((pr_res - pr_gold)**2) / sum(pr_gold**2)):.4f}%")

  
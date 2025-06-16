import numpy as np
import pybnesian as pbn
from .util_syntethic import *
import time
import glob
import os
import time
from collections import defaultdict
import natsort as ns

class loadNETs:
    def __init__(self, path):
        self.models = {}
        self.arcs = {}
        self.samples = {}
        self.groupnames = {}
        grouped_names = defaultdict(list)
        for filename in glob.glob(path):
            basename = os.path.basename(filename).split("_")[:-1]
            basename = "_".join(basename)
            
            if filename.endswith("arcs.csv"):
                arcs = pd.read_csv(filename)
                arclist = [(str(arc[0]), str(arc[1])) for arc in arcs.values]
                self.arcs[basename] = arclist
            else:
                sample = pd.read_csv(filename)
                self.samples[basename] = sample
                splitedname = basename.split("_")[0]
                grouped_names[splitedname].append(basename)

        self.groupnames = grouped_names

        for key, sample in self.samples.items():
            if key in self.arcs:
                arcs = self.arcs[key]
                nodes = list(sample.columns.values)
                model = pbn.KDENetwork(nodes=nodes, arcs=arcs)
                model.fit(sample)
                self.models[key] = model

    def get_model(self, model_name):
        return self.models[model_name]
    
    def sample(self, model_name, n, seed=1):
        if model_name in self.models:
            model = self.samples[model_name]
            sample = model.sample(n, random_state=seed)
            return sample

    def names(self, group):
        # print(self.groupnames)
        return self.groupnames

class TrainModels:
    def __init__(self, training_target:pd.DataFrame, source_list:list[pd.DataFrame], js_points:int=10000):
        self.training_target = training_target
        self.source_list = source_list
        self.nodes = training_target.columns.values
        
        js_div = pbn.TransferKDE.jensen_shannon_div(training_target, source_list, points=js_points)
        self.js_df = pd.DataFrame(js_div, columns=self.nodes)

    def RCoT_KDE(self, testdata:pd.DataFrame, transfer:bool, alpha:float=0.05):
        if transfer:
            rcot = pbn.RCoT(self.training_target, self.source_list, divergence=self.js_df, alpha=alpha)
        else:
            rcot = pbn.RCoT(self.training_target)
        
        pdag = pbn.PC().estimate(rcot, alpha=alpha)
        try:
            dag = pdag.to_dag()
        except ValueError:
            dag = pdag.to_approximate_dag()
        
        if transfer:
            model = pbn.TransferKDENetwork(dag)
            model.fit(self.training_target, source_list=self.source_list, divergence=self.js_df, use_SPBN=True)
        else:
            model = pbn.KDENetwork(dag)
            model.fit(self.training_target)
        logl = model.logl(testdata)
        return model, logl
        
    def HC_KDE  (self, testdata:pd.DataFrame, transfer:bool, kcv:int=10, **kwargs):
        pool = pbn.OperatorPool([pbn.ArcOperatorSet()])
        if transfer:
            start_model = pbn.TransferKDENetwork(nodes=self.nodes)
            score = pbn.ValidatedLikelihoodTransfer(df=self.training_target, source_list=self.source_list, divergence=self.js_df, k=kcv, use_SPBN=True)
            model = pbn.GreedyHillClimbing().estimate(pool, score, start_model, **kwargs)
            model.fit(self.training_target, source_list=self.source_list, divergence=self.js_df, use_SPBN=True)
        else:
            start_model = pbn.KDENetwork(nodes=self.nodes)
            score = pbn.ValidatedLikelihood(df=self.training_target, k=kcv)
            model = pbn.GreedyHillClimbing().estimate(pool, score, start_model, **kwargs)
            model.fit(self.training_target)
            
        logl = model.logl(testdata)
        return model, logl
    
    def structure_metrics(self, model, reference):
        nodemap = {node: i for i, node in enumerate(model.nodes())}
        hmd = hamming_distance(model.arcs(), reference.arcs(), nodemap)
        complexity = len(model.arcs())
        shd = structural_hamming_distance(model.arcs(), reference.arcs())
        return hmd, complexity, shd
    
    @staticmethod
    def measure_time(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return (*result, elapsed_time)


def append_results(res, *args):
    for key, metric in zip(res.keys(),args):
        res[key].append(metric)
    return res

def save_models(models, path, names):
    for model, name in zip(models, names):
        os.makedirs(path, exist_ok=True)
        model.save(path+f"/{name}")  # Retry without include_cpd


def rmse(logl_est, logl_true):
    """
    Compute the root mean squared error (RMSE) between two log-likelihoods.

    Parameters:
    logl_est (float): Estimated log-likelihood.
    logl_true (float): True log-likelihood.

    Returns:
    float: The RMSE between the two log-likelihoods.
    """
    return np.sqrt(np.mean((logl_est - logl_true) ** 2))

def relative_error(logl_est, logl_true):
    """
    Compute the relative error between two log-likelihoods.

    Parameters:
    logl_est (float): Estimated log-likelihood.
    logl_true (float): True log-likelihood.

    Returns:
    float: The relative error between the two log-likelihoods.
    """
    return np.mean(np.abs(logl_est - logl_true) / np.abs(logl_true))

def sumlogl(logl):
    """
    Compute the sum of log-likelihoods.

    Parameters:
    logl (numpy.ndarray): Array of log-likelihoods.

    Returns:
    float: The sum of log-likelihoods.
    """
    return np.sum(logl)


def hamming_distance(arcs1, arcs2, node_map):
    """
    Compute the Hamming distance between two graphs represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.
    num_nodes (int): Number of nodes in the graphs.

    Returns:
    int: The Hamming distance between the two graphs.
    """
    # Convert arcs to adjacency matrices
    graph1 = arcs_to_adjacency_matrix(arcs1, node_map)
    graph2 = arcs_to_adjacency_matrix(arcs2, node_map)

    # Compute the Hamming distance between the adjacency matrices
    hamming_dist = np.sum(np.abs(graph1 - graph2))

    return hamming_dist / 2

def structural_hamming_distance(arcs1, arcs2):
    """
    Compute the structural Hamming distance between two directed acyclic graphs (DAGs)
    represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.

    Returns:
    int: The structural Hamming distance between the two graphs.
    """
    # Convert lists of arcs to sets for efficient comparison
    arcs_set1 = set(arcs1)
    arcs_set2 = set(arcs2)


    hamming_dist = 0
    for arc1 in arcs1:
            if arc1 not in arcs_set2 and (arc1[1], arc1[0]) in arcs_set2: 
                hamming_dist += 1 # inverse arc 
            elif arc1 not in arcs_set2 and (arc1[1], arc1[0]) not in arcs_set2: 
                hamming_dist += 1 # removal arc 

    for arc2 in arcs2:
        if arc2 not in arcs_set1 and (arc2[1], arc2[0]) not in arcs_set1:
            hamming_dist += 1 # addition arc

    return hamming_dist


def node_type_hamming_distance(node_types, node_types_ref):
    distance = 0

    for k,v in node_types.items():
        if v != node_types_ref[k]:
            if v==pbn.FBKernelType() and node_types_ref[k]==pbn.CKDEType():
                continue
            distance += 1
    
    return distance

def arcs_to_DAG(arcs, node_map):
    """
    Convert a list of arcs to an adjacency matrix.

    Parameters:
    arcs (list of tuples): List of arcs (edges) in the graph, represented as tuples of letters.
    node_map (dict): Mapping from letters to integer node identifiers.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    num_nodes = len(node_map)
    dag_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for arc in arcs:
        # Map letters to integer node identifiers
        i, j = node_map[arc[0]], node_map[arc[1]]
        dag_matrix[i, j] = 1 # arrow from i to j  

    return dag_matrix

def average_dags(dag_list, threshold=0.5):
    """
    Compute a binary average adjacency matrix from a list of DAG adjacency matrices,
    using a threshold to decide the presence of edges.

    :param adjacency_matrices: List of numpy arrays representing adjacency matrices.
    :param threshold: Threshold to decide the presence of an edge in the final matrix.
                      The default value is 0.5.
    :return: Binary adjacency matrix as a numpy array.
    """
    # Check if the list is not empty
    if  dag_list.shape[0] == 0:
        raise ValueError("The list of adjacency matrices is empty.")

    # Sum all adjacency matrices
    summed_matrix = np.sum(dag_list, axis=0)

    # Compute the average by dividing by the number of matrices
    average_matrix = summed_matrix / dag_list.shape[0]

    # Apply the threshold to obtain a binary matrix
    binary_matrix = (average_matrix >= threshold).astype(int)

    return binary_matrix



def arcs_to_adjacency_matrix(arcs, node_map):
    """
    Convert a list of arcs to an adjacency matrix.

    Parameters:
    arcs (list of tuples): List of arcs (edges) in the graph, represented as tuples of letters.
    node_map (dict): Mapping from letters to integer node identifiers.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    num_nodes = len(node_map)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for arc in arcs:
        # Map letters to integer node identifiers
        i, j = node_map[arc[0]], node_map[arc[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # For undirected graphs

    return adj_matrix


import os
import pandas as pd
import pickle
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
import rustworkx as rx
from qiskit.compiler import transpile

def get_depth(circuit):
    return circuit.depth()

def save_to_pickle(df, filename):
    if os.path.exists(filename):
        df_old = pd.read_pickle(f'{filename}')
        df_combined = pd.concat([df_old, df], ignore_index=True)
    else:
        df_combined = df

    df_combined.to_pickle(f'{filename}')

def create_n_regular_graph(n, k):
    graph = rx.PyGraph()
    graph.add_nodes_from(range(n))
    edge_list = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edge_list.append((i, (i + j) % n, 1.0))

    if k % 2 == 1:
        for i in range(n):
            edge_list.append((i, (i + n // 2) % n, 1.0))
    graph.add_edges_from(edge_list)
    return graph

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.
 
    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("ZZ", [edge[0], edge[1]], weight))
    return pauli_list

# def cost_func_estimator(params, ansatz, hamiltonian, estimator):
#     # transform the observable defined on virtual qubits to
#     # an observable defined on all physical qubits
#     isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
 
#     pub = (ansatz, isa_hamiltonian, params)
#     job = estimator.run([pub])
 
#     results = job.result()[0]
#     cost = results.data.evs
 
#     objective_func_vals.append(cost)
 
#     return cost
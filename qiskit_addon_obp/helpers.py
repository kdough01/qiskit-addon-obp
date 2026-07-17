import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import networkx as nx
import rustworkx as rx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.compiler import transpile
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian, generate_time_evolution_circuit

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

def get_qaoa(n, k, reps=2, gamma=0.73, beta=0.42):
    """
    Generates the QAOA circuit used for testing as well as
    the list of observables needed.

    Inputs:
    - n: int - number of vertices in graph
    - k: int - degree of vertices in graph
    - gamma: rotation angle in circuit params
    - beta: rotation angle in circuit params
    """
    graph = create_n_regular_graph(n, k)
 
    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_sparse_list(max_cut_paulis, n)
    observables = [p.to_label() for p in cost_hamiltonian.paulis]

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    init_params = {}
    for param in circuit.parameters:
        if param.name.startswith('γ'):
            init_params[param] = gamma
        elif param.name.startswith('β'):
            init_params[param] = beta
    circuit = circuit.assign_parameters(init_params)
    circuit = transpile(circuit, basis_gates=['cx', 'rz', 'rx', 'h'])

    return circuit, observables

def get_heisenberg(depth):

    """
    Generates the heisenberg circuit used for testing
    """
    coupling_map = CouplingMap.from_heavy_hex(3, bidirectional=False)
    reduced_coupling_map = coupling_map.reduce([0, 13, 1, 14, 10, 16, 5, 12, 8, 18])

    hamiltonian = generate_xyz_hamiltonian(
                                            reduced_coupling_map,
                                            coupling_constants=(np.pi / 8, np.pi / 4, np.pi / 2),
                                            ext_magnetic_field=(np.pi / 3, np.pi / 6, np.pi / 9),
                                        )
    circuit = generate_time_evolution_circuit(
                                                hamiltonian,
                                                time=0.1,
                                                synthesis=LieTrotter(reps=depth),
                                            )
    
    return circuit

def make_heisenberg_hamiltonian(
    num_qubits: int,
    J: float = 1.0,
) -> SparsePauliOp:
    """
    Build the isotropic Heisenberg XXX Hamiltonian as in the Qiskit algorithms tutorial.

    Parameters
    ----------
    num_qubits : int
    J : float
        Uniform exchange coupling. Tutorial uses J=1.0.
    """
    terms = []
    for i in range(num_qubits - 1):
        for pauli_char in ["XX", "YY", "ZZ"]:
            pauli = "I" * i + pauli_char + "I" * (num_qubits - i - 2)
            terms.append((pauli, J))
    return SparsePauliOp.from_list(terms).simplify()

def generate_heisenberg_circuit(
    num_qubits: int = 6,
    time: float = 1.6,
    trotter_reps: int = 4,
    trotter_order: int = 1,
    J: float = 1.0,
    initial_state: str = "neel",  # "neel", "hadamard", or None
) -> Tuple[QuantumCircuit, Dict[str, Any]]:

    hamiltonian = make_heisenberg_hamiltonian(num_qubits, J=J)
    synthesis = (
        LieTrotter(reps=trotter_reps)
        if trotter_order == 1
        else SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    )
    evo_circuit = generate_time_evolution_circuit(hamiltonian, synthesis=synthesis, time=time)

    qc = QuantumCircuit(num_qubits)
    if initial_state == "neel":
        for i in range(0, num_qubits, 2):
            qc.x(i)
    elif initial_state == "hadamard":
        for i in range(num_qubits):
            qc.h(i)

    qc.compose(evo_circuit, inplace=True)

    observable_str = "I" * (num_qubits - 1) + "Z"

    metadata = dict(
        model         = "Heisenberg XXX",
        reference     = "Qiskit Algorithms tutorial: Quantum Real Time Evolution (13_trotterQRTE)",
        num_qubits    = num_qubits,
        time          = time,
        trotter_reps  = trotter_reps,
        trotter_order = trotter_order,
        J             = J,
        initial_state = initial_state,
        observable    = observable_str,
        n_pauli_terms = len(hamiltonian),
    )
    return qc, metadata

def get_heisenberg_40(depth):
    """
    Generates a 40-qubit XYZ Heisenberg circuit on a heavy-hex topology.
    """
    coupling_map = CouplingMap.from_heavy_hex(5, bidirectional=False)
    
    edges = [(int(a), int(b)) for a, b in coupling_map.get_edges()]
    G = nx.Graph()
    G.add_edges_from(edges)
    
    connected_40_nodes = list(nx.bfs_tree(G, source=0).nodes())[:40]
    
    reduced_coupling_map = coupling_map.reduce(connected_40_nodes)

    hamiltonian = generate_xyz_hamiltonian(
        reduced_coupling_map,
        coupling_constants=(np.pi / 8, np.pi / 4, np.pi / 2),
        ext_magnetic_field=(np.pi / 3, np.pi / 6, np.pi / 9),
    )
    
    circuit = generate_time_evolution_circuit(
        hamiltonian,
        time=0.1,
        synthesis=LieTrotter(reps=depth),
    )
    
    return circuit
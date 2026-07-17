import os

from qiskit_addon_obp.utils.truncating import setup_budget
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import qiskit.quantum_info as qi
from qiskit_aer import AerSimulator
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit_addon_utils.slicing import slice_by_gate_types
from qiskit_addon_utils.slicing import combine_slices
from qiskit.synthesis import LieTrotter
# from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.backpropagation import backpropagate
from qiskit.primitives import StatevectorEstimator
import time
from qiskit import qpy
from qiskit.qasm2 import dumps
import json
import pickle
import random
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator
import pandas as pd
from qiskit.circuit.library import QAOAAnsatz
import rustworkx as rx
from qiskit_addon_obp.helpers import create_n_regular_graph, build_max_cut_paulis, get_heisenberg, get_qaoa, get_depth, save_to_pickle, generate_heisenberg_circuit
from qiskit.compiler import transpile
import math
from qiskit_aer.primitives import EstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeLimaV2
from qiskit_aer.noise import NoiseModel
from qiskit_addon_obp.benchmarks.benchmark_suite import load_qasmbench

def process_backpropagated_circuit(obs, circuit, target_depth, max_qwc_groups, max_error_per_slice, coeff_truncate, pauli_truncate, truncation_weight):
    """
    Process the backpropagated circuit and convert it to proper format for shadow estimation

    Inputs:
    - obs: str - observable in string format, e.g. "ZZIIIIIIII"
    - circuit: QuantumCircuit - original quantum circuit to be backpropagated
    - target_depth: int - desired maximum depth of the backpropagated circuit
    - max_qwc_groups: int - maximum number of qubit-wise commuting groups allowed in the backpropagated observable
    - max_error_per_slice: float - maximum allowable error per slice during backpropagation
    - coeff_truncate: bool - whether to enable coefficient truncation
    - pauli_truncate: bool - whether to enable Pauli string truncation
    - truncation_weight: float - Pauli-weight used for the pauli truncation scheme

    Outputs:
    - data_dict: dict - dictionary containing information about the backpropagated circuit and observable
    - circuit: QuantumCircuit - original quantum circuit
    """
    errors = []
    times = []
    circuits = []
    circuit_slices = []
    sparse_paulis = []
    
    slices = slice_by_gate_types(circuit)

    op_budget = OperatorBudget(max_paulis=None, max_qwc_groups=None)
    truncation_error_budget = setup_budget(max_error_per_slice=max_error_per_slice)

    observable = SparsePauliOp(obs)
    start = time.time()
    bp_obs_trunc, remaining_slices_trunc, metadata = backpropagate(
        observable, slices, target_depth=target_depth, truncation_error_budget=truncation_error_budget, coeff_truncate=coeff_truncate, pauli_truncate=pauli_truncate, truncation_weight=truncation_weight
    )
    end = time.time()
    print(f'Backpropagation took {end - start} seconds')

    bp_circuit_trunc = combine_slices(remaining_slices_trunc, include_barriers=True)

    data_dict = {
        "backpropagated slices": metadata.num_backpropagated_slices,
        "number of terms": len(bp_obs_trunc.paulis),
        "number of groups": len(bp_obs_trunc.group_commuting(qubit_wise=True)),
        "max error": max_error_per_slice,
        "error is bounded by": metadata.accumulated_error(0),
        "bp_obs": bp_obs_trunc,  
        "bp_circuit": bp_circuit_trunc,
        "obs": obs,
        "circuit": circuit,
        "bp_circuit_depth": bp_circuit_trunc.depth()
    }

    circuits.append(bp_circuit_trunc)
    sparse_paulis.append(bp_obs_trunc)

    return data_dict, circuit

def shot_count_per_group(N, coefs, err):
    """
    Inputs:
    - N: int - number of commuting groups
    - coefs - max coefficient in the QWC group
    - err - precision of total measurements
    """
    c = coefs
    g = N**2 * c / (err ** 2)
    return g

def shot_allocation(S, v_i, v_sum):
    """
    - S - total number of shots
    - v_i - squared maximum weight of the commuting group
    - v_sum - sum of all of the squared maximum weights of the commuting groups
    """
    s_i = int(math.floor(S * v_i / v_sum))
    return s_i if s_i > 0 else 1

def estimate_circuit(circuit, observable, bp_circuit_trunc, bp_obs_trunc, shots_per_group=None, commuting_groups=None, obp_shots=10000, noisy=False):
    """
    This simulates two quantum circuits using Qiskit AerSimulator: one initial circuit, and one 
    backpropagated circuit. It takes in an observable and estimates the expectation value of
    the observable using a number of shots specified by the obp_shots parameter.

    Inputs:
    - circuit: QuantumCircuit - the original quantum circuit
    - observable: SparsePauliOp - the observable to estimate
    - bp_circuit_trunc: QuantumCircuit - the backpropagated circuit
    - bp_obs_trunc: SparsePauliOp - the backpropagated observable
    - obp_shots: int - number of shots to use for the estimation

    Outputs:
    - result_bp_trunc: float - estimated expectation value from the backpropagated circuit
    - result_exact: float - exact expectation value from the original circuit
    - result_bp_trunc_state: float - exact expectation value from the backpropagated circuit
    """
    errors = []

    if noisy:
        print("USING NOISY SIM")
        backend = AerSimulator(noise_model=NoiseModel.from_backend(FakeLimaV2()))
    else:
        backend = AerSimulator()
    
    # pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    
    # bp_circuit_trunc_isa = pm.run(bp_circuit_trunc)
    # bp_obs_trunc_isa = bp_obs_trunc.apply_layout(bp_circuit_trunc_isa.layout)

    bp_circuit_trunc_isa = bp_circuit_trunc
    bp_obs_trunc_isa = bp_obs_trunc

    if circuit.num_qubits > 20:
        state_vector_estimator = EstimatorV2(
            options={"backend_options": {"method": "matrix_product_state", "matrix_product_state_max_bond_dimension": 16}}
        )
    else:
        state_vector_estimator = StatevectorEstimator()

    result_exact = (state_vector_estimator.run([(circuit, observable)]).result()[0]).data.evs.item()
    result_bp_trunc_state = (state_vector_estimator.run([(bp_circuit_trunc_isa, bp_obs_trunc_isa)]).result()[0].data.evs.item())

    if shots_per_group is None:
        estimator = Estimator(backend, options={"default_shots": obp_shots})
        result_bp_trunc = estimator.run([(bp_circuit_trunc_isa, bp_obs_trunc_isa)]).result()[0].data.evs.item()
    else:
        group_estimates = []
        group_weights = []
        group_errors = []
        for shot, group in zip(shots_per_group, commuting_groups):
            estimator = Estimator(backend, options={"default_shots": shot})
            group_result_bp_trunc = estimator.run([(bp_circuit_trunc_isa, group)]).result()[0].data.evs.item()
            result_bp_trunc_state = (state_vector_estimator.run([(bp_circuit_trunc_isa, group)]).result()[0].data.evs.item())
            weight = sum(abs(coeff) for coeff in group.coeffs)
            group_estimates.append(group_result_bp_trunc)
            group_weights.append(weight)
            group_errors.append(abs(result_bp_trunc_state - group_result_bp_trunc))

        # print(group_errors)
        result_bp_trunc = 0
        for est, w, s in zip(group_estimates, group_weights, shots_per_group):
            if s > 0:
                result_bp_trunc += (est * w)
        # result_bp_trunc = result_bp_trunc / sum(group_weights)

    error = result_exact - result_bp_trunc

    errors.append(error)

    return result_bp_trunc, result_exact, result_bp_trunc_state

def save_circuits_to_text(all_circuits_list, observables, filename="circuits.txt"):
    """
    Save all circuits to a single text file for easy viewing
    
    Args:
        all_circuits_list: List of circuit lists, one for each observable
        observables: List of observable strings  
        filename: Output text filename
    """
    with open(filename, 'w') as f:
        for obs_idx, circuits in enumerate(all_circuits_list):
            obs = observables[obs_idx]
            f.write(f"\n{'='*60}\n")
            f.write(f"OBSERVABLE: {obs}\n")
            f.write(f"{'='*60}\n\n")
            
            for circuit_idx, circuit in enumerate(circuits):
                f.write(f"Circuit {circuit_idx}:\n")
                f.write(f"{'-'*40}\n")
                f.write(str(circuit.draw(output='text')))
                f.write(f"\n{'-'*40}\n\n")
    
    print(f"All circuits saved to {filename}")

def convert_observables(sparse_pauli_ops):
    """
    Convert a list of SparsePauliOp observables into a specific string format.
    
    Inputs:
    - sparse_pauli_ops: list of SparsePauliOp - list of observables to convert
    
    Outputs:
    - result_str: str - formatted string representation of the observables
    """
    if not sparse_pauli_ops:
        return ""
    
    # Get system size from the first SparsePauliOp
    first_pauli_string = str(sparse_pauli_ops[0].paulis[0])
    system_size = len(first_pauli_string)
    
    result_lines = [str(system_size)]
    
    for sparse_pauli_op in sparse_pauli_ops:
        for pauli_string in sparse_pauli_op.paulis:
            pauli_str = str(pauli_string)
            k_local = 0
            operators = []
            for qubit_idx, pauli in enumerate(pauli_str):
                if pauli != 'I':
                    operators.append((pauli, qubit_idx))
                    k_local += 1
            
            if k_local > 0:
                line_parts = [str(k_local)]
                for pauli, qubit_idx in operators:
                    line_parts.extend([pauli, str(qubit_idx)])
                result_lines.append(' '.join(line_parts))
    
    return '\n'.join(result_lines)

def convert_observables_for_many(sparse_pauli_ops):
    """
    Convert a list of SparsePauliOp observables into a specific string format without system size

    Inputs:
    - sparse_pauli_ops: list of SparsePauliOp - list of observables to convert

    Outputs:
    - result_str: str - formatted string representation of the observables
    """
    if not sparse_pauli_ops:
        return ""
    
    # Get system size from the first SparsePauliOp
    first_pauli_string = str(sparse_pauli_ops[0].paulis[0])
    system_size = len(first_pauli_string)
    
    result_lines = []
    
    for sparse_pauli_op in sparse_pauli_ops:
        for pauli_string in sparse_pauli_op.paulis:
            pauli_str = str(pauli_string)
            k_local = 0
            operators = []
            for qubit_idx, pauli in enumerate(pauli_str):
                if pauli != 'I':
                    operators.append((pauli, qubit_idx))
                    k_local += 1
            
            if k_local > 0:
                line_parts = [str(k_local)]
                for pauli, qubit_idx in operators:
                    line_parts.extend([pauli, str(qubit_idx)])
                result_lines.append(' '.join(line_parts))
    
    return '\n'.join(result_lines)

def run_backpropagation(obs, target_depth,max_error=0.01, max_error_increment=0.005, operator_budget=8, operator_budget_increment=2, coeff_truncate=False, pauli_truncate=False, noisy=False):
    """
    Run backpropagation for a single observable over a range of error thresholds and operator budgets.

    Inputs:
    - obs: str - observable in string format, e.g. "ZZIIIIIIII"
    - target_depth: int - desired maximum depth of the backpropagated circuit
    - max_error: float - maximum allowable error per slice during backpropagation
    - max_error_increment: float - increment for the error threshold in each iteration
    - operator_budget: int - maximum number of qubit-wise commuting groups allowed in the backpropagated observable
    - operator_budget_increment: int - increment for the operator budget in each iteration
    - coeff_truncate: bool - whether to enable coefficient truncation
    - pauli_truncate: bool - whether to enable Pauli string truncation

    Outputs:
    - data_dict_list: list of dicts for one observable
    """
    data_dict_list = []
    obp_time = 0.0
    err = 0.0
    while err <= max_error:
        budget = 1
        while budget <= operator_budget:

            start = time.time()
            data_dict, circuit = process_backpropagated_circuit(
                                                        obs,
                                                        target_depth=target_depth,
                                                        max_qwc_groups=budget,
                                                        max_error_per_slice=err, 
                                                        coeff_truncate=coeff_truncate,
                                                        pauli_truncate=pauli_truncate
                                                        )

            if data_dict["bp_circuit"] is None:
                print("skipping")
                budget += 1
                continue

            end = time.time()
            data_dict["expectation value with truncation"], data_dict["exact expectation value"], data_dict['result_bp_trunc_state'] = estimate_circuit(
                                                                                                                    circuit=circuit, 
                                                                                                                    observable=obs, 
                                                                                                                    bp_circuit_trunc=data_dict["bp_circuit"],
                                                                                                                    bp_obs_trunc=data_dict["bp_obs"],
                                                                                                                    obp_shots=10000,
                                                                                                                    noisy=noisy
                                                                                                                    )
            data_dict_list.append(data_dict)
            obp_time += (end - start)

            budget += operator_budget_increment

        err += max_error_increment

    data_dict_list = [item for item in data_dict_list if item["bp_circuit"]!=None]

    return data_dict_list

def obp_protocol(
        observable,
        circuit,
        target_depth,
        optimal_shot_allocation=False,
        max_qwc_groups=4,
        max_error_per_slice=0.01,
        obp_shots=10000,
        coeff_truncate=False,
        pauli_truncate=False,
        truncation_weight=7,
        noisy=False
        ):
    """
    Run the OBP protocol for a single observable and quantum circuit.

    Inputs:
    - observable: str - observable in string format, e.g. "ZZIIIIIIII
    - circuit: QuantumCircuit - original quantum circuit to be backpropagated
    - target_depth: int - desired maximum depth of the backpropagated circuit
    - max_qwc_groups: int - maximum number of qubit-wise commuting groups allowed in the backpropagated observable
    - max_error_per_slice: float - maximum allowable error per slice during backpropagation
    - obp_shots: int - number of shots to use for the estimation
    - coeff_truncate: bool - whether to enable coefficient truncation
    - pauli_truncate: bool - whether to enable Pauli string truncation
    - truncation_weight: float - Pauli-weight used for the pauli truncation scheme
    
    Outputs:
    - all_data_dict_lists: list of dicts - list containing information about the backpropagated circuit and observable
    - pauli_strings_list: list of lists - list of Pauli strings for each backpropagated observable
    - pauli_coeffs_list: list of lists - list of coefficients for each backpropagated observable
    """
    ### OBP ###
    observables = [observable]

    all_data_dict_lists = []
        
    data_dict_list, circuit = process_backpropagated_circuit(
                                                            observables[0],
                                                            circuit,
                                                            target_depth=target_depth,
                                                            max_qwc_groups=max_qwc_groups,
                                                            max_error_per_slice=max_error_per_slice,
                                                            coeff_truncate=coeff_truncate,
                                                            pauli_truncate=pauli_truncate,
                                                            truncation_weight=truncation_weight
                                                            )
    if data_dict_list["bp_circuit"] is None:
        print("skipping")
        # return [], [], []
    else:
        if optimal_shot_allocation:
            # TODO: Each group needs to have at least 1 shot
            print("Using optimal shot allocation...")
            N = len(data_dict_list["bp_obs"].group_commuting(qubit_wise=True))
            coef_sqr = []
            for group in data_dict_list["bp_obs"].group_commuting(qubit_wise=True):
                coef_sqr.append(max(group.coeffs.real ** 2))
            shots = []
            coef_sqr_sum = sum(coef_sqr)
            for coef in coef_sqr:
                shots.append(shot_allocation(obp_shots, coef, coef_sqr_sum))

            # only increase the max value if the total number of shots is less than the allowed number of total shots, but we will usually be overestimating
            # because we set the number of shots equal to 1 if any group is 0
            if obp_shots > sum(shots):
                max_val = max(shots)
                max_idx = shots.index(max_val)
                shots[max_idx] = max_val + obp_shots - sum(shots)

            total_shots = sum(shots)
            # print(shots)
            data_dict_list["bp_exp_shots"], data_dict_list["exact_exp"], data_dict_list['bp_exp_state'] = estimate_circuit(
                                                                                    circuit=circuit, 
                                                                                    observable=data_dict_list['obs'], 
                                                                                    bp_circuit_trunc=data_dict_list["bp_circuit"],
                                                                                    bp_obs_trunc=data_dict_list["bp_obs"],
                                                                                    obp_shots=obp_shots,
                                                                                    shots_per_group=shots,
                                                                                    commuting_groups=data_dict_list["bp_obs"].group_commuting(qubit_wise=True),
                                                                                    noisy=noisy
                                                                                    )

        else:
            total_shots = obp_shots
            data_dict_list["bp_exp_shots"], data_dict_list["exact_exp"], data_dict_list['bp_exp_state'] = estimate_circuit(
                                                                                    circuit=circuit, 
                                                                                    observable=data_dict_list['obs'], 
                                                                                    bp_circuit_trunc=data_dict_list["bp_circuit"],
                                                                                    bp_obs_trunc=data_dict_list["bp_obs"],
                                                                                    obp_shots=obp_shots,
                                                                                    noisy=noisy
                                                                                    )
    all_data_dict_lists.append(data_dict_list)

    pauli_strings_list = []
    pauli_coeffs_list = []
    for item in all_data_dict_lists:
        sparse_pauli_op = item["bp_obs"]
        strings = [str(pauli) for pauli in sparse_pauli_op.paulis]
        pauli_strings_list.append(strings)
            
        coeffs = [complex(coeff) for coeff in sparse_pauli_op.coeffs]
        pauli_coeffs_list.append(coeffs)

    sample_observables = []
    for item in all_data_dict_lists:
        sample_observables.append(item["bp_obs"])

    # with open("obp_obs.txt", "a") as f:
    #     f.write(convert_observables_for_many(sample_observables))
    #     f.write('\n')

    return all_data_dict_lists, pauli_strings_list, pauli_coeffs_list, total_shots

def run_obp(observables, target_depth, circuit_type, optimal_shot_allocation, budget=4, max_error_per_slice=0.01, depth=5, obp_shots=10000, n=5, k=3, noisy=False, pauli_truncate=False):
    """
    Run the OBP protocol for a list of observables on a generated quantum circuit.

    Inputs:
    - observables: list - takes in a list of observables, even if only entering one observable, must be of list form
    - which_circuit: bool - if True, the original circuit will be used, otherwise, if False, the backpropagated circuits will be used
    - target_depth: int - desired maximum depth of the backpropagated circuit
    - budget: int - maximum number of qubit-wise commuting groups allowed in the backpropagated observable
    - max_error_per_slice: float - maximum allowable error per slice during backpropagation
    - depth: int - number of Trotter steps of the original quantum circuit
    - obp_shots: int - number of shots to use for the estimation

    Outputs:
    - all_data_dict_lists: list of dicts - list containing information about the backpropagated circuit and observable
    - pauli_strings_list: list of lists - list of Pauli strings for each backpropagated observable
    - pauli_coeffs_list: list of lists - list of coefficients for each backpropagated observable
    """

    if circuit_type == "qaoa":
        circuit, observables = get_qaoa(n, k, reps=depth)
    elif circuit_type == "heisenberg_10":
        circuit = get_heisenberg(depth)
    elif circuit_type == "heisenberg":
        circuit, _ = generate_heisenberg_circuit(num_qubits=40, time=0.1, trotter_reps=2)
    elif circuit_type == "qasm_bench":
        circuit = load_qasmbench("ising_n26", "medium")

    print(f"Circuit Depth: {circuit.depth()}")

    with open("obp_obs.txt", "a") as f:
        f.write(str(len(observables[0])))
        f.write('\n')

    all_data_dict_lists = []
    all_pauli_strings_list = []
    all_pauli_coeffs_list = []
    for obs in observables:
        new_data, pauli_strings_list, pauli_coeffs_list, total_shots = obp_protocol(
                                                                        obs,
                                                                        circuit=circuit,
                                                                        target_depth=target_depth,
                                                                        optimal_shot_allocation=optimal_shot_allocation,
                                                                        max_qwc_groups=budget,
                                                                        max_error_per_slice=max_error_per_slice,
                                                                        obp_shots=obp_shots,
                                                                        noisy=noisy,
                                                                        pauli_truncate=pauli_truncate)
        new_data[0]['obp_shots'] = total_shots
        all_data_dict_lists += new_data

    return all_data_dict_lists

def main():

    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data3'))
    filename = f'{data_path}/obp_bp10_pauli_trunc_obs9.pkl'

    #1_800_000    10**6
    start_shots = 1
    max_shots = 100000
    # 100000

    data_list = []
    all_data_dict_lists = []
    data_list = []
    shots = start_shots
    one = "I" * 19 + "ZZ" + "I" * 19
    two = "I" * 19 + "ZIZ" + "I" * 18
    three = "I" * 38 + "XX"

    # n = 18
    # observables = [
    #     "Z" + "I"*(n-1),           # Z on qubit 0 (MSB)
    #     "I"*(n-1) + "Z",           # Z on qubit 17 (LSB)
    #     "I"*8 + "Z" + "I"*9,       # Z near the middle (qubit 8)
        
    #     "Z" + "I"*(n-2) + "Z",     # ZZ on first and last qubit
    #     "I"*8 + "ZZ" + "I"*8,      # Middle ZZ correlator
    # ]

    observables = ["IIIIIIIIIIIIIZIIIIIIIIIIII",
                   "IIIIIIIIIIIIIZZIIIIIIIIIII",
                   "IIIIIIIIIIIIIXIIIIIIIIIIII"]

    while shots <= max_shots:

        print(f"Shots: {shots}")
        new_data = run_obp(observables=#observables,#[one, two, three
                                    ["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                                     "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                                     "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

                                    #  "IZZIIIIIII", "IZXIIIIIII", "IZYIIIIIII",
                                    #  "IXXIIIIIII", "IXZIIIIIII", "IXYIIIIIII",
                                    #  "IYYIIIIIII", "IYZIIIIIII", "IYXIIIIIII",

                                    #  "IIZZIIIIII", "IIZXIIIIII", "IIZYIIIIII",
                                    #  "IIXXIIIIII", "IIXZIIIIII", "IIXYIIIIII",
                                    #  "IIYYIIIIII", "IIYZIIIIII", "IIYXIIIIII",

                                    #  "IIIZZIIIII", "IIIZXIIIII", "IIIZYIIIII",
                                    #  "IIIXXIIIII", "IIIXZIIIII", "IIIXYIIIII",
                                    #  "IIIYYIIIII", "IIIYZIIIII", "IIIYXIIIII"
                                    ],
                            budget=10,
                            target_depth=10,
                            circuit_type="heisenberg_10",
                            optimal_shot_allocation=False,
                            max_error_per_slice=0.0001,
                            depth=10, # for heis this is 9*depth, for QAOA this sets the reps
                            obp_shots=shots,
                            n=10,
                            k=6,
                            noisy=False,
                            pauli_truncate=True
                        )
        
        shots *= 10
        data_list.append(new_data)

        all_data_dict_lists += new_data

    df = pd.DataFrame(all_data_dict_lists)

    df['circuit_depth'] = df['circuit'].apply(get_depth)
    df['bp_circuit_depth'] = df['bp_circuit'].apply(get_depth)

    df['abs_bp_error_shots'] = abs(df['exact_exp'] - df['bp_exp_shots'])
    df['bp_error_shots'] = df['exact_exp'] - df['bp_exp_shots']

    df['abs_bp_error_state'] = abs(df['exact_exp'] - df['bp_exp_state'])
    df['bp_error_state'] = df['exact_exp'] - df['bp_exp_state']

    save_to_pickle(df, filename)

    with open('measurements.txt', 'w') as f:
        pass

    with open('obp_obs.txt', 'w') as f:
        pass
    
if __name__ == "__main__":
    main()
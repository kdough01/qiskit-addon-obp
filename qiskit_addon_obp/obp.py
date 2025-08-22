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

def process_backpropagated_circuit(obs, circuit, target_depth, max_qwc_groups, max_error_per_slice, coeff_truncate, pauli_truncate, truncation_weight):
    """
    Process the backpropagated circuit and convert it to proper format for shadow estimation
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

def estimate_circuit(circuit, observable, bp_circuit_trunc, bp_obs_trunc, obp_shots=10000):
    """
    This simulates two quantum circuits using Qiskit AerSimulator: one initial circuit, and one 
    backpropagated circuit. It takes in an observable and estimates the expectation value of
    the observable using a number of shots specified by the obp_shots parameter.
    """
    errors = []

    backend = AerSimulator()
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    
    bp_circuit_trunc_isa = pm.run(bp_circuit_trunc)
    bp_obs_trunc_isa = bp_obs_trunc.apply_layout(bp_circuit_trunc_isa.layout)

    estimator = Estimator(backend, options={"default_shots": obp_shots})
    state_vector_estimator = StatevectorEstimator()

    result_exact = (state_vector_estimator.run([(circuit, observable)]).result()[0]).data.evs.item()
    result_bp_trunc = estimator.run([(bp_circuit_trunc_isa, bp_obs_trunc_isa)]).result()[0].data.evs.item()
    result_bp_trunc_state = (state_vector_estimator.run([(bp_circuit_trunc_isa, bp_obs_trunc_isa)]).result()[0].data.evs.item())

    # print(f"length of commuting obs: {len(bp_obs_trunc.group_commuting(qubit_wise=True))}")
    # print(f"length of obs: {len(bp_obs_trunc.paulis)}")

    error = result_exact - result_bp_trunc

    errors.append(error)

    return result_bp_trunc, result_exact, result_bp_trunc_state

def export_for_old_qiskit(circuits, observables, slices, filename):
    circuit_data = []
    for i, circuit in enumerate(circuits):
        qasm_str = dumps(circuit)
        
        circuit_info = {
            'num_qubits': circuit.num_qubits,
            'num_clbits': circuit.num_clbits,
            'name': circuit.name,
            'qasm': qasm_str
        }
        circuit_data.append(circuit_info)

    with open(filename, 'w') as f:
        json.dump(circuit_data, f, indent=2)

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

def run_backpropagation(obs, target_depth,max_error=0.01, max_error_increment=0.005, operator_budget=8, operator_budget_increment=2, coeff_truncate=False, pauli_truncate=False):
    """
    Inputs:

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
                                                                                                                    obp_shots=10000
                                                                                                                    )
            data_dict_list.append(data_dict)
            obp_time += (end - start)

            budget += operator_budget_increment

        err += max_error_increment

    data_dict_list = [item for item in data_dict_list if item["bp_circuit"]!=None]

    return data_dict_list

def obp_protocol(observable, circuit, target_depth, max_qwc_groups=4, max_error_per_slice=0.01, obp_shots=10000, coeff_truncate=False, pauli_truncate=False, truncation_weight=7):
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
        data_dict_list["bp_exp_shots"], data_dict_list["exact_exp"], data_dict_list['bp_exp_state'] = estimate_circuit(
                                                                                circuit=circuit, 
                                                                                observable=data_dict_list['obs'], 
                                                                                bp_circuit_trunc=data_dict_list["bp_circuit"],
                                                                                bp_obs_trunc=data_dict_list["bp_obs"],
                                                                                obp_shots=obp_shots
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

    #flatten pauli list and remove duplicates

    sample_observables = []
    for item in all_data_dict_lists:
        sample_observables.append(item["bp_obs"])

    with open("obp_obs.txt", "a") as f:
        f.write(convert_observables_for_many(sample_observables))
        f.write('\n')

    return all_data_dict_lists, pauli_strings_list, pauli_coeffs_list

def run_obp(observables, target_depth, budget=4, max_error_per_slice=0.01, depth=5, obp_shots=10000):
    """
    Inputs:
    - observables: list - takes in a list of observables, even if only entering one observable, must be of list form
    - which_circuit: bool - if True, the original circuit will be used, otherwise, if False, the backpropagated circuits will be used
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
    print(f"Circuit Depth: {circuit.depth()}")

    with open("obp_obs.txt", "a") as f:
        f.write(str(len(observables[0])))
        f.write('\n')

    all_data_dict_lists = []
    all_pauli_strings_list = []
    all_pauli_coeffs_list = []
    for obs in observables:
        new_data, pauli_strings_list, pauli_coeffs_list = obp_protocol(obs, circuit=circuit, target_depth=target_depth, max_qwc_groups=budget, max_error_per_slice=max_error_per_slice, obp_shots=obp_shots)
        new_data[0]['obp_shots'] = obp_shots
        all_data_dict_lists += new_data

    return all_data_dict_lists

def get_depth(circuit):
    return circuit.depth()

def save_to_pickle(df, filename):
    if os.path.exists(filename):
        df_old = pd.read_pickle(f'{filename}')
        df_combined = pd.concat([df_old, df], ignore_index=True)
    else:
        df_combined = df

    df_combined.to_pickle(f'{filename}')

def main():

    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data5'))
    filename = f'{data_path}/obp_bp10_obs36.pkl'

    start_shots = 1
    max_shots = 10000

    data_list = []
    all_data_dict_lists = []
    data_list = []
    shots = start_shots

    while shots <= max_shots:

        print(f"Shots: {shots}")
        new_data = run_obp(observables=["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                                     "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                                     "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

                                     "IZZIIIIIII", "IZXIIIIIII", "IZYIIIIIII",
                                     "IXXIIIIIII", "IXZIIIIIII", "IXYIIIIIII",
                                     "IYYIIIIIII", "IYZIIIIIII", "IYXIIIIIII",

                                     "IIZZIIIIII", "IIZXIIIIII", "IIZYIIIIII",
                                     "IIXXIIIIII", "IIXZIIIIII", "IIXYIIIIII",
                                     "IIYYIIIIII", "IIYZIIIIII", "IIYXIIIIII",

                                     "IIIZZIIIII", "IIIZXIIIII", "IIIZYIIIII",
                                     "IIIXXIIIII", "IIIXZIIIII", "IIIXYIIIII",
                                     "IIIYYIIIII", "IIIYZIIIII", "IIIYXIIIII"
                                     ],
                            budget=10,
                            target_depth=10,
                            max_error_per_slice=0.0001,
                            depth=10,
                            obp_shots=shots
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
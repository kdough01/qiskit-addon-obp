"""
This is intended to run many backpropagated circuits all at once starting from some
initial circuit and creating one measurement file with all of this data.
"""

import os
import sys
import time

from qiskit_addon_obp.obp import convert_observables_for_many, process_backpropagated_circuit, estimate_circuit, obp_protocol, convert_observables_for_many
from qiskit_addon_obp.classical_shadows import derandomized_classical_shadow, estimate_exp, generate_shadow_measurements, save_measurements_to_file, generate_statevector_shadow_measurements
import numpy as np
from qiskit.synthesis import LieTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
import json
from qiskit.primitives import StatevectorEstimator
import pandas as pd
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
import inspect
from collections import defaultdict
import random
import math
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Depth
# from qiskit_addon_cutting import cut_wires, partition_circuit_qubits, generate_cutting_experiments, reconstruct_expectation_value

def measurement(circuit, budget, file='obp_obs.txt', measurements_per_observable=100, shots_per_measurement=100, noisy=False):

    with open(file) as f:
        content = f.readlines()
    system_size = int(content[0])

    all_observables = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        # NOTE: JUST ADDED THIS IN
        """
        In theory, gettign rid of observables that are already measured should
        still give us the same length measurement scheme because derandomization should
        have been getting rid of them
        """
        if one_observable not in all_observables:
            all_observables.append(one_observable)

        """
        NOTE: This is the output before we added the two above lines
        Beginning Derandomization...
        Total number of measurements: 186450
            Derandomization took 4624.3725690841675 seconds
            Derandomization produced a measurement scheme 15383 long
        Generating Normal Measurements...
            Number of identity only rounds: 0
            Number of processed rounds: 15383
            Generating measurements took 126.86695694923401 seconds
        Saving measurements to file...
        Total Time: 4893.248570919037 seconds

        NOTE: This is the output after we added the two above lines
        Beginning Derandomization...
        Total number of measurements: 81520
            Derandomization took 2077.159033060074 seconds
            Derandomization produced a measurement scheme 15362 long
        Generating Normal Measurements...
            Number of identity only rounds: 0
            Number of processed rounds: 15362
            Generating measurements took 126.19540023803711 seconds
        Saving measurements to file...
        Total Time: 2346.233466863632 seconds

        NOTE: They generate a measurement scheme nearly identical in length. After adding the
        two lines, though it takes less than half the time. It also seems to be more accurate
        though that could just be the shadow variation.
        """

    print(f"Beginning Derandomization...")
    start = time.time()
    measurement_procedure = derandomized_classical_shadow(all_observables, int(measurements_per_observable), system_size)
    end = time.time()
    print(f"    Derandomization took {end - start} seconds")
    print(f"    Derandomization produced a measurement scheme {len(measurement_procedure)} long")
    # measurement_procedure = randomized_classical_shadow(measurements_per_observable, system_size=system_size)
    caller_name = inspect.currentframe().f_back.f_code.co_name
    if caller_name == "run_many" or caller_name == "adaptive" or caller_name == "run_pauli_prop":
        print(f"Generating Normal Measurements...")
        start = time.time()
        measurements = generate_shadow_measurements(measurement_scheme=measurement_procedure, budget=budget, quantum_state_circuit=circuit, shots_per_measurement=shots_per_measurement, noisy=noisy)
        end = time.time()
    elif caller_name == "run_many_state_vector":
        print(f"Generating StateVector Measurements...")
        start = time.time()
        measurements = generate_statevector_shadow_measurements(measurement_scheme=measurement_procedure, quantum_state_circuit=circuit)
        end = time.time()
    print(f"    Generating measurements took {end - start} seconds")
    print(f"Saving measurements to file...")
    save_measurements_to_file(measurements, filename='measurements.txt')

    return measurements

def reconstruct_pauli_string():
    reconstructed_strings = []

    with open('obp_obs.txt', 'r') as f:
        content = f.readlines()
    
    op_length = int(content[0])
    for line in content[1:]:
        elements = line.split()
        num_operators = int(elements[0])
        operators = elements[1:]
        
        full_string = ['I'] * op_length
        
        for i in range(num_operators):
            operator_type = operators[2*i]
            position = int(operators[2*i + 1])
            full_string[position] = operator_type
        
        reconstructed_strings.append(''.join(full_string))

    return reconstructed_strings

def shadow_estimates_dict(sample_observables, file='obp_obs.txt'):
    with open(file) as f:
        content = f.readlines()
    
    with open('measurements.txt') as f:
        measurements = f.readlines()

    full_measurement = []
    for line in measurements[1:]:
        single_meaurement = []
        for pauli_XYZ, outcome in zip(line.split(" ")[0::2], line.split(" ")[1::2]):
            single_meaurement.append((pauli_XYZ, float(outcome)))
        full_measurement.append(single_meaurement)

    variances = []
    shadows = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        sum_product, cnt_match, products = estimate_exp(full_measurement, one_observable)
        if cnt_match > 1:
            var = np.var(products, ddof=1)
        else: var = 0.0
        shadows.append(sum_product / cnt_match)
        variances.append(var)
        # print(sum_product / cnt_match)

    obs_shad_dict = {}
    for observable, shadow, var in zip(sample_observables, shadows, variances):
        obs_shad_dict[observable] = {"mean": shadow, "variance": float(var)}

    return obs_shad_dict

def measure_again(pauli_strings_list, pauli_coeffs_list, obs_shad_dict, error_thresh):
    remeasure = []

    for p_string_list, p_coeff_list in zip(pauli_strings_list, pauli_coeffs_list):
        for p_string, p_coeff in zip(p_string_list, p_coeff_list):
            eps = (p_coeff.real ** 2) * obs_shad_dict[p_string]['variance']
            if eps > error_thresh and p_coeff.real > 1e-3:
                # print(p_coeff.real)
                remeasure.append(p_string)
            else:
                print(p_coeff.real)

    return remeasure

def og_shadow_estimates(all_data_dict_lists, pauli_strings_list, pauli_coeffs_list, obs_shad_dict):
    shadow_estimates = []

    for p_string_list, p_coeff_list in zip(pauli_strings_list, pauli_coeffs_list):
        shadow_estimate = 0.0
        for p_string, p_coeff in zip(p_string_list, p_coeff_list):
            if p_coeff:
                shadow_estimate += p_coeff * obs_shad_dict[p_string]['mean']
        shadow_estimates.append(shadow_estimate.real)

    all_data_dict_lists['total_exp'] = shadow_estimate.real
    # all_data_dict_lists['total_var'] = variances

    return all_data_dict_lists

def og_shadow_estimates_state_vector(all_data_dict_lists, pauli_strings_list, pauli_coeffs_list, obs_shad_dict):
    shadow_estimates = []

    for p_string_list, p_coeff_list in zip(pauli_strings_list, pauli_coeffs_list):
        shadow_estimate = 0.0
        for p_string, p_coeff in zip(p_string_list, p_coeff_list):
            if p_coeff:
                shadow_estimate += p_coeff * obs_shad_dict[p_string]['mean']
        shadow_estimates.append(shadow_estimate.real)

    all_data_dict_lists['total_exp'] = shadow_estimate.real
    # all_data_dict_lists['total_var'] = variances

    return all_data_dict_lists

def obs_shad_exact(circuit, obs_shad_dict):
    """
    This simulates two quantum circuits using Qiskit AerSimulator: one initial circuit, and one 
    backpropagated circuit. It takes in an observable and estimates the expectation value of
    the observable using a number of shots specified by the obp_shots parameter.
    """

    output = []

    for idx, (obs, val) in enumerate(obs_shad_dict.items()):
        estimate = val["mean"]
        variance = val["variance"]

        state_vector_estimator = StatevectorEstimator()

        result_exact = (state_vector_estimator.run([(circuit, obs)]).result()[0]).data.evs.item()

        error = result_exact - estimate

        output.append({
            'obs': obs,
            'estimate': estimate,
            'exact': result_exact,
            'error': error,
            'var': variance
        })

    return output

# def perform_circuit_cutting(circuit, target_depth):
#     """
#     Cuts the circuit at a specified target depth and returns two subcircuits.

#     Args:
#         circuit (QuantumCircuit): the circuit to be cut
#         target_depth (int): the depth at which to cut the circuit
#     Returns:
#         tuple: two subcircuits resulting from the cut
#     """

#     dag = circuit_to_dag(circuit)
#     layers_by_depth = list(dag.layers())

#     target_layer = layers_by_depth[target_depth]["graph"].op_nodes()

#     cut_circuit = circuit.copy()

#     for qubit in circuit.qubits:
#         for node in target_layer:
#             if qubit in node.qargs:
#                 cut_wire(cut_circuit, qubit)
#                 break

#     subcircuits = partition_circuit(cut_circuit)
#     return subcircuits[0], subcircuits[1]

def convert_obs_to_form(observables):
    result_lines = []
    for pauli_str in observables:
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

def adaptive(
        observables,
        target_depth,
        budget=4,
        max_error_per_slice=0.01,
        measurements_per_observable=10,
        shots_per_measurement=100,
        depth=5,
        noisy=False,
        obp_shots=10000,
        coeff_truncate=False,
        pauli_truncate=False,
        truncation_weight=7,
):
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
        new_data, pauli_strings_list, pauli_coeffs_list = obp_protocol(
                                                                    obs, 
                                                                    circuit=circuit, 
                                                                    target_depth=target_depth, 
                                                                    max_qwc_groups=budget, 
                                                                    max_error_per_slice=max_error_per_slice, 
                                                                    obp_shots=obp_shots, 
                                                                    coeff_truncate=coeff_truncate,
                                                                    pauli_truncate=pauli_truncate,
                                                                    truncation_weight=truncation_weight
                                                                    )
        all_data_dict_lists += new_data
        all_pauli_strings_list.append(pauli_strings_list)
        all_pauli_coeffs_list.append(pauli_coeffs_list)

    measurement(circuit=all_data_dict_lists[0]['bp_circuit'], budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)

    with open('measurements.txt', 'r') as f:
        content = f.readlines()
        all_observables = defaultdict(lambda: defaultdict(list))
        for line in content:
            for idx, (pauli_XYZ, position) in enumerate(zip(line.split(" ")[0::2], line.split(" ")[1::2])):
                all_observables[idx][pauli_XYZ].append(int(position))

    ran_sample = random.sample(list(all_observables.items()), 50)
    pauli_operators = ['X', 'Y', 'Z']
    new_measurements = ''
    for new_meas in range(10):
        new_line = '\n'
        for idx in range(10):
            pauli = random.choice(list(ran_sample[idx].keys()))
            synth_meas = random.choice(ran_sample[idx][pauli]) #random.choice([-1, 1])

            new_line += pauli + ' ' + str(synth_meas) + ' '
        new_measurements += new_line

    with open('measurements.txt', 'a') as f:
        f.write(new_measurements)

    sample_observables = reconstruct_pauli_string()

    obs_shad_dict = shadow_estimates_dict(sample_observables)
    obs_shad_df = obs_shad_exact(circuit, obs_shad_dict)

    # for obs in obs_shad_dict:
    #     if obs in obs_shad_dict:
    #         obs_shad_dict[obs]['mean'] = obs_shad_dict[obs]
    
    for obs_idx in range(len(observables)):
        og_shadow_estimates(all_data_dict_lists[obs_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)
        all_data_dict_lists[obs_idx]['obs_shad_dict'] = obs_shad_df

    return all_data_dict_lists

def run_circuit_cutting(
        observables,
        target_depth,
        budget=4,
        max_error_per_slice=0.01,
        measurements_per_observable=10,
        shots_per_measurement=100,
        depth=5,
        which_circuit=False,
        noisy=False,
        obp_shots=10000,
        coeff_truncate=False,
        pauli_truncate=False,
        truncation_weight=7
        ):
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
        new_data, pauli_strings_list, pauli_coeffs_list = obp_protocol(
                                                                    obs, 
                                                                    circuit=circuit, 
                                                                    target_depth=target_depth, 
                                                                    max_qwc_groups=budget, 
                                                                    max_error_per_slice=max_error_per_slice, 
                                                                    obp_shots=obp_shots, 
                                                                    coeff_truncate=coeff_truncate,
                                                                    pauli_truncate=pauli_truncate,
                                                                    truncation_weight=truncation_weight
                                                                    )
        all_data_dict_lists += new_data
        all_pauli_strings_list.append(pauli_strings_list)
        all_pauli_coeffs_list.append(pauli_coeffs_list)

    # first_half, second_half = perform_circuit_cutting(circuit, target_depth)

    # subcircuits = [first_half, second_half]
    # experiments = generate_cutting_experiments(subcircuits)

    # sim = AerSimulator()
    # results = [sim.run(exp).result() for exp in experiments]

    # exp_val = reconstruct_expectation_value(results, observables)
    # print("Expectation value:", exp_val)

def run_many(
        observables,
        target_depth,
        budget=4,
        max_error_per_slice=0.01,
        measurements_per_observable=10,
        shots_per_measurement=100,
        depth=5,
        which_circuit=False,
        noisy=False,
        obp_shots=10000,
        coeff_truncate=False,
        pauli_truncate=False,
        truncation_weight=7,
        ):
    """
    Inputs:
    - observables: list - takes in a list of observables, even if only entering one observable, must be of list form
    - which_circuit: bool - if True, the original circuit will be used, otherwise, if False, the backpropagated circuits will be used
    """
    # observables = [
    #     "ZIIIIIIIII",
    #     "XIIIIIIIII",
    #     "ZXZIIIIIII"
    # ]

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
        new_data, pauli_strings_list, pauli_coeffs_list = obp_protocol(
                                                                    obs, 
                                                                    circuit=circuit, 
                                                                    target_depth=target_depth, 
                                                                    max_qwc_groups=budget, 
                                                                    max_error_per_slice=max_error_per_slice, 
                                                                    obp_shots=obp_shots, 
                                                                    coeff_truncate=coeff_truncate,
                                                                    pauli_truncate=pauli_truncate,
                                                                    truncation_weight=truncation_weight
                                                                    )
        all_data_dict_lists += new_data
        all_pauli_strings_list.append(pauli_strings_list)
        all_pauli_coeffs_list.append(pauli_coeffs_list)

    if which_circuit:
        measurement(circuit=circuit, budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)
    else:
        measurement(circuit=all_data_dict_lists[0]['bp_circuit'], budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)

    sample_observables = reconstruct_pauli_string()

    # It uses the same shadow estimates for all the backpropagated circuits, which is why we need to make sure the
    # backpropagated circuits are to the same depth
    obs_shad_dict = shadow_estimates_dict(sample_observables)
    obs_shad_df = obs_shad_exact(circuit, obs_shad_dict)
    with open('obs_shad_dict.json', 'w') as f:
        json.dump(obs_shad_df, f)

    for obs_idx in range(len(observables)):
        og_shadow_estimates(all_data_dict_lists[obs_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)
        all_data_dict_lists[obs_idx]['obs_shad_dict'] = obs_shad_df

    # for obs_idx in range(len(observables)):
    #     for data_idx in range(len(all_data_dict_lists)):
    #         og_shadow_estimates(all_data_dict_lists[data_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)

    with open("obp_tests.txt", "a") as f:
        for obs_idx, data in enumerate(all_data_dict_lists):
            f.write(f"\n=== Observable {all_data_dict_lists[obs_idx]['obs']} ===\n")
            print(data, file=f)

    return all_data_dict_lists

def run_many_state_vector(
        observables,
        target_depth,
        budget=4,
        max_error_per_slice=0.01,
        measurements_per_observable=10,
        shots_per_measurement=100,
        depth=5,
        which_circuit=False,
        noisy=False,
        obp_shots=10000,
        coeff_truncate=False,
        pauli_truncate=False,
        truncation_weight=7,
        ):
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
        new_data, pauli_strings_list, pauli_coeffs_list = obp_protocol(
                                                                    obs, 
                                                                    circuit=circuit, 
                                                                    target_depth=target_depth, 
                                                                    max_qwc_groups=budget, 
                                                                    max_error_per_slice=max_error_per_slice, 
                                                                    obp_shots=obp_shots, 
                                                                    coeff_truncate=coeff_truncate,
                                                                    pauli_truncate=pauli_truncate,
                                                                    truncation_weight=truncation_weight
                                                                    )
        all_data_dict_lists += new_data
        all_pauli_strings_list.append(pauli_strings_list)
        all_pauli_coeffs_list.append(pauli_coeffs_list)

    if which_circuit:
        measurement(circuit=circuit, budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)
    else:
        measurement(circuit=all_data_dict_lists[0]['bp_circuit'], budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)

    # print(obs_shad_dict)

    sample_observables = reconstruct_pauli_string()

    # It uses the same shadow estimates for all the backpropagated circuits, which is why we need to make sure the
    # backpropagated circuits are to the same depth
    obs_shad_dict = shadow_estimates_dict(sample_observables)
    obs_shad_df = obs_shad_exact(circuit, obs_shad_dict)
    with open('obs_shad_dict.json', 'w') as f:
        json.dump(obs_shad_df, f)

    for obs_idx in range(len(observables)):
        og_shadow_estimates_state_vector(all_data_dict_lists[obs_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)
        # all_data_dict_lists[obs_idx]['obs_shad_dict'] = obs_shad_df

    # for obs_idx in range(len(observables)):
    #     for data_idx in range(len(all_data_dict_lists)):
    #         og_shadow_estimates(all_data_dict_lists[data_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)

    with open("obp_tests.txt", "a") as f:
        for obs_idx, data in enumerate(all_data_dict_lists):
            f.write(f"\n=== Observable {all_data_dict_lists[obs_idx]['obs']} ===\n")
            print(data, file=f)

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

def normal_state_vector():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'state_vector_data'))
    filename = f'{data_path}/coef-trunc_init90_bp5_obs3.pkl'

    meas_per_obs = 1
    data_list = []
    start = time.time()
    new_data = run_many_state_vector(observables=["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                                    #  "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                                    #  "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

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
                        target_depth=5,
                        max_error_per_slice=0.0001,
                        measurements_per_observable=meas_per_obs,
                        shots_per_measurement=1,
                        depth=10,
                        which_circuit=False,
                        noisy=False,
                        obp_shots=10000,
                        coeff_truncate=True,
                        pauli_truncate=False,
                        truncation_weight=7)
    end = time.time()
    
    data_list += new_data

    df = pd.DataFrame(data_list)

    df['num_meas'] = meas_per_obs
    df['circuit_depth'] = df['circuit'].apply(get_depth)
    df['bp_circuit_depth'] = df['bp_circuit'].apply(get_depth)


    df['abs_bp_error_shots'] = abs(df['exact_exp'] - df['bp_exp_shots'])
    df['bp_error_shots'] = df['exact_exp'] - df['bp_exp_shots']

    df['abs_bp_error_state'] = abs(df['exact_exp'] - df['bp_exp_state'])
    df['bp_error_state'] = df['exact_exp'] - df['bp_exp_state']

    df['abs_total_error'] = abs(df['exact_exp'] - df['total_exp'])

    df['total_error'] = df['exact_exp'] - df['total_exp']

    save_to_pickle(df, filename)

    with open('measurements.txt', 'w') as f:
        pass

    with open('obp_obs.txt', 'w') as f:
        pass

    print(f"Total Time: {end - start} seconds")

def normal():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data4'))
    filename = f'{data_path}/coef-trunc_init90_bp5_obs18.pkl'
    #coef-trunc_init90_bp10_obs9

    data_list = []
    start = time.time()
    new_data = run_many(observables=["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                                     "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                                     "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

                                     "IZZIIIIIII", "IZXIIIIIII", "IZYIIIIIII",
                                     "IXXIIIIIII", "IXZIIIIIII", "IXYIIIIIII",
                                     "IYYIIIIIII", "IYZIIIIIII", "IYXIIIIIII",

                                    #  "IIZZIIIIII", "IIZXIIIIII", "IIZYIIIIII",
                                    #  "IIXXIIIIII", "IIXZIIIIII", "IIXYIIIIII",
                                    #  "IIYYIIIIII", "IIYZIIIIII", "IIYXIIIIII",

                                    #  "IIIZZIIIII", "IIIZXIIIII", "IIIZYIIIII",
                                    #  "IIIXXIIIII", "IIIXZIIIII", "IIIXYIIIII",
                                    #  "IIIYYIIIII", "IIIYZIIIII", "IIIYXIIIII"
                                     ],
                        budget=10,
                        target_depth=5,
                        max_error_per_slice=0.0001,
                        measurements_per_observable=500,
                        shots_per_measurement=1,
                        depth=10,
                        which_circuit=False,
                        noisy=False,
                        obp_shots=10000,
                        coeff_truncate=True,
                        pauli_truncate=False,
                        truncation_weight=7)
    end = time.time()
    
    if new_data:
        with open('measurements.txt', 'r') as f:
            file = f.readlines()
        data_list += new_data

    df = pd.DataFrame(data_list)

    df['num_meas'] = len(file) - 1
    df['circuit_depth'] = df['circuit'].apply(get_depth)
    df['bp_circuit_depth'] = df['bp_circuit'].apply(get_depth)


    df['abs_bp_error_shots'] = abs(df['exact_exp'] - df['bp_exp_shots'])
    df['bp_error_shots'] = df['exact_exp'] - df['bp_exp_shots']

    df['abs_bp_error_state'] = abs(df['exact_exp'] - df['bp_exp_state'])
    df['bp_error_state'] = df['exact_exp'] - df['bp_exp_state']

    df['abs_total_error'] = abs(df['exact_exp'] - df['total_exp'])

    df['total_error'] = df['exact_exp'] - df['total_exp']

    save_to_pickle(df, filename)

    with open("measurements.txt", "rb") as f:
        file_size = os.path.getsize('measurements.txt')
        print(f"Measurement file size: {file_size} bytes")

    with open("obp_obs.txt", "rb") as f:
        file_size = os.path.getsize('obp_obs.txt')
        print(f"Observable file size: {file_size} bytes")

    with open('obp_obs.txt', 'w') as f:
        pass

    with open('measurements.txt', 'w') as f:
        pass

    print(f"Total Time: {end - start} seconds")

def new():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data_adaptive'))
    filename = f'{data_path}/test.pkl'

    obp_shots = 1000
    data_list = []
    start = time.time()
    new_data = adaptive(observables=["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                                    #  "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                                    #  "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

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
                        target_depth=70,
                        max_error_per_slice=0.0001,
                        measurements_per_observable=10,
                        shots_per_measurement=1,
                        depth=10,
                        noisy=False,
                        obp_shots=obp_shots,
                        coeff_truncate=True,
                        pauli_truncate=False,
                        truncation_weight=7)
    end = time.time()
    
    if new_data:
        with open('measurements.txt', 'r') as f:
            file = f.readlines()
            file_length = len(file) - 1
        data_list += new_data

        with open('meas_again.txt' , 'r') as fi:
            new_file = fi.readlines()
            new_file_length = len(new_file) - 1

    df = pd.DataFrame(data_list)

    print(f'lenght of file is {file_length}')
    df['num_meas'] = file_length #+ (new_file_length) * obp_shots
    print(df['num_meas'])
    df['circuit_depth'] = df['circuit'].apply(get_depth)
    df['bp_circuit_depth'] = df['bp_circuit'].apply(get_depth)


    df['abs_bp_error_shots'] = abs(df['exact_exp'] - df['bp_exp_shots'])
    df['bp_error_shots'] = df['exact_exp'] - df['bp_exp_shots']

    df['abs_bp_error_state'] = abs(df['exact_exp'] - df['bp_exp_state'])
    df['bp_error_state'] = df['exact_exp'] - df['bp_exp_state']

    df['abs_total_error'] = abs(df['exact_exp'] - df['total_exp'])

    df['total_error'] = df['exact_exp'] - df['total_exp']

    save_to_pickle(df, filename)

    with open('measurements.txt', 'w') as f:
        pass

    with open('obp_obs.txt', 'w') as f:
        pass

    with open('meas_again.txt', 'w') as f:
        pass

    print(f"Total Time: {end - start} seconds")

def main():
    normal()

if __name__ == '__main__':
    main()
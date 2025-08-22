#
# This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
#  "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#
import os
import random
import math
import random
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeLimaV2
from qiskit_aer.noise import NoiseModel
from qiskit import ClassicalRegister, QuantumCircuit
import numpy as np
import json
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit_addon_obp.obp import convert_observables
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian, generate_time_evolution_circuit
import pandas as pd
from qiskit.primitives import StatevectorEstimator
from collections import defaultdict

def generate_observables(file, system_size = 10):
    observable_file = open(file, 'w')

    print(system_size, file = observable_file)

    for i in range(system_size - 1):
        for j in range(system_size - 1):
            if j == i or j == i + 1 or j+1 == i: continue
            print("4 Y {} Y {} X {} X {}".format(i, i+1, j, j+1), file = observable_file)

    for i in range(system_size - 1):
        for j in range(system_size):
            if j == i or j == i + 1: continue
            for j2 in range(system_size):
                if j2 == i or j2 == i + 1 or j2 == j: continue
                print("4 X {} X {} Z {} Z {}".format(i, i+1, j, j2), file = observable_file)

    for i in range(system_size - 1):
        for j in range(system_size):
            if j == i or j == i + 1: continue
            print("3 X {} X {} Z {}".format(i, i+1, j), file = observable_file)

def randomized_classical_shadow(num_total_measurements, system_size):
    #
    # Implementation of the randomized classical shadow
    #
    #    num_total_measurements: int for the total number of measurement rounds
    #    system_size: int for how many qubits in the quantum system
    #
    measurement_procedure = []
    for t in range(num_total_measurements):
        single_round_measurement = [random.choice(["X", "Y", "Z"]) for i in range(system_size)]
        measurement_procedure.append(single_round_measurement)
    return measurement_procedure

def derandomized_classical_shadow(all_observables, num_of_measurements_per_observable, system_size, weight=None):
    #
    # Implementation of the derandomized classical shadow
    #
    #     all_observables: a list of Pauli observables, each Pauli observable is a list of tuple
    #                      of the form ("X", position) or ("Y", position) or ("Z", position)
    #     num_of_measurements_per_observable: int for the number of measurement for each observable
    #     system_size: int for how many qubits in the quantum system
    #     weight: None or a list of coefficients for each observable
    #             None -- neglect this parameter
    #             a list -- modify the number of measurements for each observable by the corresponding weight
    #
    if weight is None:
        weight = [1.0] * len(all_observables)
    assert(len(weight) == len(all_observables))

    sum_log_value = 0
    sum_cnt = 0

    def cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift = 0):
        eta = 0.9 # a hyperparameter subject to change
        nu = 1 - math.exp(-eta / 2)

        nonlocal sum_log_value
        nonlocal sum_cnt

        cost = 0
        for i, zipitem in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round)):
            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                continue

            if system_size < matches_needed:
                V = eta / 2 * measurement_so_far
            else:
                V = eta / 2 * measurement_so_far - math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-V / weight[i] - shift)

            sum_log_value += V / weight[i]
            sum_cnt += 1

        return cost

    def match_up(qubit_i, dice_roll_pauli, single_observable):
        for pauli, pos in single_observable:
            if pos != qubit_i:
                continue
            else:
                if pauli != dice_roll_pauli:
                    return -1
                else:
                    return 1
        return 0

    num_of_measurements_so_far = [0] * len(all_observables)
    measurement_procedure = []
    print(f"Total number of measurements: {num_of_measurements_per_observable * len(all_observables)}")
    for repetition in range(num_of_measurements_per_observable * len(all_observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [len(P) for P in all_observables]
        single_round_measurement = []

        shift = sum_log_value / sum_cnt if sum_cnt > 0 else 0
        sum_log_value = 0.0
        sum_cnt = 0

        for qubit_i in range(system_size):
            cost_of_outcomes = dict([("X", 0), ("Y", 0), ("Z", 0)])

            for dice_roll_pauli in ["X", "Y", "Z"]:
                # Assume the dice rollout to be "dice_roll_pauli"
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z

                cost_of_outcomes[dice_roll_pauli] = cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift=shift)

                # Revert the dice roll
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] -= 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] += 1 # match up one Pauli X/Y/Z

            for dice_roll_pauli in ["X", "Y", "Z"]:
                if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                    continue
                # The best dice roll outcome will come to this line
                single_round_measurement.append(dice_roll_pauli)
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z
                break

        measurement_procedure.append(single_round_measurement)

        for i, single_observable in enumerate(all_observables):
            if num_of_matches_needed_in_this_round[i] == 0: # finished measuring all qubits
                num_of_measurements_so_far[i] += 1

        success = 0
        for i, single_observable in enumerate(all_observables):
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                success += 1

        if success == len(all_observables):
            break

    return measurement_procedure

def estimate_exp(full_measurement, one_observable):

    sum_product, cnt_match = 0, 0
    products = []

    for single_measurement in full_measurement:
        not_match = 0
        product = 1

        for pauli_XYZ, position in one_observable:
            if pauli_XYZ != single_measurement[position][0]:
                not_match = 1
                break
            product *= single_measurement[position][1]

        if not_match == 1: continue

        sum_product += product
        cnt_match += 1
        products.append(product)

    return sum_product, cnt_match, products

def generate_statevector_shadow_measurements(measurement_scheme, quantum_state_circuit):
    """
    Generate shot-noise-free shadow measurements by computing exact single-qubit expectation values
    for the derandomized measurement scheme. This gives you the measurement outcomes that would
    feed into the classical shadow estimator, but without any shot noise.
    
    Returns measurement data in the same format as the noisy version, but with exact values.
    """
    
    system_size = len(measurement_scheme[0])
    estimator = StatevectorEstimator()
    result_lines = [f"{system_size}"]
    
    # Store all single-qubit expectation values we compute
    observable_outcomes = defaultdict(list)
    
    print(f"Computing shot-noise-free shadow measurements for {len(measurement_scheme)} rounds...")
    
    for round_idx, measurement_round in enumerate(measurement_scheme):
        if round_idx % 100 == 0:
            print(f"  Processing round {round_idx}/{len(measurement_scheme)}")
        
        line = []
        
        # For each qubit in this measurement round, get the exact expectation value
        for qubit_idx, pauli_op in enumerate(measurement_round):
            if pauli_op in ['X', 'Y', 'Z']:
                # Create single-qubit Pauli observable
                pauli_string = 'I' * qubit_idx + pauli_op + 'I' * (system_size - qubit_idx - 1)
                observable = SparsePauliOp.from_list([(pauli_string, 1.0)])
                
                # Get exact expectation value (this eliminates shot noise)
                job = estimator.run([(quantum_state_circuit, observable)])
                result = job.result()
                expectation_value = result[0].data.evs.real
                
                # Store for shadow reconstruction
                observable_key = (qubit_idx, pauli_op)
                observable_outcomes[observable_key].append(expectation_value)
                
                line.append(f"{pauli_op} {expectation_value}")
            elif pauli_op == 'I':
                # Identity measurements
                line.append(f"I 1.0")
        
        result_lines.append(' '.join(line))
    
    return '\n'.join(result_lines)

def generate_shadow_measurements(measurement_scheme, budget, quantum_state_circuit, shots_per_measurement, noisy=False):
    """
    This generates shadow measurement data using the measurement scheme generated by the randomization/derandomization procedures.

    Inputs:
    - noisy: bool - set to False for non-noisy data, set to True for noisy data
    """
    system_size = len(measurement_scheme[0])
    if noisy:
        print(f'    Using noisy simulator')
        backend = FakeLimaV2()
        noise_model = NoiseModel.from_backend(backend)
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()
    result_lines = [f"{system_size}"]

    identity_only_rounds = 0
    processed_rounds = 0

    observable_outcomes = defaultdict(list)

    for measurement_round in measurement_scheme:
        non_identity_ops = [op for op in measurement_round if op in ['X', 'Y', 'Z']]
        
        if not non_identity_ops:
            identity_only_rounds += 1

        # print('depth before adding gates: ', quantum_state_circuit.depth())
        measurement_circuit = convert_pauli(quantum_state_circuit, measurement_round)
        # print('depth after adding gates: ', measurement_circuit.depth())

        job = simulator.run(measurement_circuit, shots=shots_per_measurement)
        result = job.result()
        counts = result.get_counts(measurement_circuit)

        # metadata = {
        #     "budget": budget,
        #     "measurement_round": measurement_round,
        #     "counts": counts,
        #     "circuit": qasm3_dumps(quantum_state_circuit)
        # }
        # with open('counts.jsonl', 'a') as f:
        #     json.dump(metadata, f)
        #     f.write('\n')

        for bitstring, count in counts.items():
            for _ in range(count):
                line = []
                bit_idx = 0
                for qubit_idx, pauli_op in enumerate(measurement_round):
                    if pauli_op in ['X', 'Y', 'Z']:
                        bit = bitstring[-(bit_idx + 1)]
                        eig = 1 if bit == '0' else -1
                        observable = (qubit_idx, pauli_op)
                        observable_outcomes[observable].append(eig)
                        line.append(f"{pauli_op} {eig}")
                        bit_idx += 1
                result_lines.append(' '.join(line))

        processed_rounds += 1

    print(f'    Number of identity only rounds: {identity_only_rounds}')
    print(f'    Number of processed rounds: {processed_rounds}')

    return '\n'.join(result_lines)

def convert_pauli(state_circuit, pauli_string):
    """
    
    """
    circuit = state_circuit.copy()
    
    num_measurements = sum(1 for pauli in pauli_string if pauli != 'I')
    
    if circuit.num_clbits < num_measurements:
        circuit.add_register(ClassicalRegister(num_measurements - circuit.num_clbits))

    for qubit_idx, pauli_op in enumerate(pauli_string):
        if pauli_op == 'X':
            circuit.ry(-np.pi/2, qubit_idx)
        elif pauli_op == 'Y':
            circuit.rx(np.pi/2, qubit_idx)

    clbit_idx = 0
    for qubit_idx, pauli_op in enumerate(pauli_string):
        if pauli_op != 'I':
            circuit.measure(qubit_idx, clbit_idx)
            clbit_idx += 1
    
    return circuit

def save_measurements_to_file(measurements, filename='measurements.txt'):
    """Save measurement data to file."""
    with open(filename, 'a') as f:
        f.write(measurements)
    # print(f"Measurements saved to '{filename}'")

def run_shadow(sample_observables, measurements_per_observable=10, shots_per_measurement=10, depth=5):
    observables = [SparsePauliOp(obs) for obs in sample_observables]
    with open("obp_obs.txt", "a") as f:
        f.write(convert_observables(observables))
        f.write('\n')

    with open('obp_obs.txt') as f:
        content = f.readlines()
    system_size = int(content[0])

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

    all_observables = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        all_observables.append(one_observable)

    # print(all_observables)
    measurement_procedure = derandomized_classical_shadow(all_observables, int(measurements_per_observable), system_size)
    # measurement_procedure = randomized_classical_shadow(measurements_per_observable, system_size=system_size)
    measurements = generate_shadow_measurements(
                                                measurement_scheme=measurement_procedure, 
                                                budget=10,
                                                quantum_state_circuit=circuit,
                                                shots_per_measurement=shots_per_measurement,
                                                noisy=False
                                            )
    save_measurements_to_file(measurements, filename='measurements.txt')

    with open('measurements.txt') as f:
        measurements = f.readlines()

    num_meas = len(measurements) - 1

    full_measurement = []
    for line in measurements[1:]:
        single_meaurement = []
        for pauli_XYZ, outcome in zip(line.split(" ")[0::2], line.split(" ")[1::2]):
            single_meaurement.append((pauli_XYZ, int(outcome)))
        full_measurement.append(single_meaurement)

    shadows = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        sum_product, cnt_match, products = estimate_exp(full_measurement, one_observable)
        shadows.append(sum_product / cnt_match)
        # print(sum_product / cnt_match)

    i = 0
    obs_dict = {}
    for one_obs, estimate in zip(content[1:], shadows):
        state_vector_estimator = StatevectorEstimator()
        result_exact = (state_vector_estimator.run([(circuit, sample_observables[i])]).result()[0]).data.evs.item()
        obs_dict[sample_observables[i]] = (estimate, result_exact)
        i += 1
        # print(f"Observable: {one_obs.strip()}, Estimate: {estimate}")

    return obs_dict, num_meas

def calculate_shadow_sample_complexity(df, num_terms, df_column, epsilon, confidence_level=0.95):
    """
    Calculate the required number of shadow copies for classical shadow tomography
    
    df: pandas DataFrame with observable data
    df_column: string of df column
    epsilon: desired additive error
    confidence_level: success probability (default 0.95)
    """
    
    if df_column in df.columns:
        max_shadow_norm_squared = (df[df_column].max())**2
    
    else:
        raise ValueError(f"DataFrame must contain column {df_column}")
    
    M = num_terms
    C = 34
    delta = 1 - confidence_level
    N = C * np.log(M) * max_shadow_norm_squared / (epsilon**2)
    N = int(np.ceil(N))

    return N

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
    """
    The observables will be generated from the OBP code
    """
    observables=observables=[
                            "ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
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
                            ]

    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data5'))
    filename = f'{data_path}/shad_res_36.pkl'

    start_meas = 25000
    max_meas = 25000

    data_list = []
    all_data_dict_lists = []
    data_list = []
    meas = start_meas
    meas_list = []

    while meas <= max_meas:

        print(f"Measurements: {meas}")
        new_data, num_meas = run_shadow(
                    sample_observables=observables,
                    measurements_per_observable=meas,
                    shots_per_measurement=1,
                    depth=10
                )
        
        meas *= 10
        
        all_data_dict_lists.append(new_data)
        meas_list.append(num_meas)

        with open('measurements.txt', 'w') as f:
            pass

        with open('obp_obs.txt', 'w') as f:
            pass

    all_data_rows = []

    meas_idx = 0

    for data_dict in all_data_dict_lists:
        for obs, (shad_exp, exact_val) in data_dict.items():
            row = {
                "shad_exp": shad_exp,
                "exact_val": exact_val,
                "shad_error": exact_val - shad_exp,
                "abs_shad_error": abs(shad_exp - exact_val),
                "num_meas": meas_list[meas_idx]
            }
            all_data_rows.append(row)
        meas_idx += 1

    df = pd.DataFrame(all_data_rows)

    save_to_pickle(df, filename)

if __name__ == "__main__":
    main()
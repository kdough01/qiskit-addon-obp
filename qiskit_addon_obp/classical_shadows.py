#
# This code (generate_observables, randomized_classical_shadow, derandomized_classical_shadow, estimate_exp)
# is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
#  "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#
import os
import math
import random
import pickle
import numpy as np
import pandas as pd

from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeLimaV2
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2

from qiskit_addon_obp.obp import convert_observables
from qiskit_addon_obp.helpers import save_to_pickle, get_heisenberg, get_qaoa
from helpers import generate_heisenberg_circuit
from qiskit_addon_obp.benchmarks.benchmark_suite import load_qasmbench

def generate_observables(file, system_size = 10):
    """
    Function from predicting-quantum-properties repository
    """
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
    """
    Function from predicting-quantum-properties repository
    """
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
    """
    Function from predicting-quantum-properties repository
    """
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
    """
    Function adapted from predicting-quantum-properties repository.
    Estimate the expectation value of one observable from the full measurement data

    Inputs:
    - full_measurement: list of list of tuple - the full measurement data, each inner list is a single measurement
                        each tuple in the inner list is of the form ("X", outcome) or ("Y", outcome) or ("Z", outcome)
    - one_observable: list of tuple - the observable to be estimated, of the form ("X", position) or ("Y", position) or ("Z", position)

    Outputs:
    - sum_product: int - the sum of the products of the measurement outcomes that match the observable
    - cnt_match: int - the number of measurements that match the observable
    - products: list of int - the list of products of the measurement outcomes that match the observable
    """

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

def generate_shadow_measurements(measurement_scheme, budget, quantum_state_circuit, shots_per_measurement=100, noisy=False):
    
    system_size = len(measurement_scheme[0])
    
    if noisy:
        simulator = AerSimulator(noise_model=NoiseModel.from_backend(FakeLimaV2()), method="matrix_product_state")
    else:
        simulator = AerSimulator(method="matrix_product_state")

    all_outcomes = []
    
    print(f"    Generating {len(measurement_scheme)} measurement rounds × {shots_per_measurement} shots...")

    for round_idx, measurement_round in enumerate(measurement_scheme):
        non_identity_ops = [op for op in measurement_round if op in ['X', 'Y', 'Z']]
        if not non_identity_ops:
            continue

        measurement_circuit = convert_pauli(quantum_state_circuit, measurement_round)
        
        job = simulator.run(measurement_circuit, shots=shots_per_measurement)
        result = job.result()
        counts = result.get_counts()

        for bitstring, count in counts.items():
            outcomes = []
            bit_idx = 0
            for pauli_op in measurement_round:
                if pauli_op in ['X', 'Y', 'Z']:
                    bit = bitstring[-(bit_idx + 1)]
                    eig = 1 if bit == '0' else -1
                    outcomes.append((pauli_op, eig))
                    bit_idx += 1
            # Store count times (more efficient than repeating)
            all_outcomes.extend([outcomes] * count)

    print(f"    Total shadow snapshots generated: {len(all_outcomes):,}")
    
    save_measurements_to_file(all_outcomes, 'measurements.pkl')
    return all_outcomes

def convert_pauli(state_circuit, pauli_string):
    """
    Perform single-qubit Pauli rotations on X and Y Pauli strings and add measurements.
    Assumes Z measurements are done in the computational basis.

    Inputs:
    - state_circuit: QuantumCircuit - circuit that prepares the quantum state to be measured
    - pauli_string: list of str - list of 'I', 'X', 'Y', 'Z' specifying the Pauli measurement on each qubit

    Outputs:
    - circuit: QuantumCircuit - modified circuit with basis change and measurements added
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

def save_measurements_to_file(measurements, filename='measurements.pkl'):
    """Much faster - Save as binary pickle instead of huge text file"""
    with open(filename, 'wb') as f:
        pickle.dump(measurements, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(measurements):,} measurement outcomes to {filename}")

def load_measurements(filename='measurements.pkl'):
    """Fast loading"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data):,} shadow snapshots")
    return data

def run_shadow(sample_observables, circuit_type, n, k, measurements_per_observable=10, shots_per_measurement=10, depth=5, noisy=False):
    """
    Run derandomized shadow tomography on a quantum system and estimate expectation values of given observables
    """

    if circuit_type == "qaoa":
        circuit, sample_observables = get_qaoa(n, k, reps=depth)

        observables = [SparsePauliOp(obs) for obs in sample_observables]
        with open("obp_obs.txt", "a") as f:
            f.write(convert_observables(observables))
            f.write('\n')

    elif circuit_type == "heisenberg_10":
        circuit = get_heisenberg(depth)

        observables = [SparsePauliOp(obs) for obs in sample_observables]
        with open("obp_obs.txt", "a") as f:
            f.write(convert_observables(observables))
            f.write('\n')

    elif circuit_type == "heisenberg":
        circuit, _ = generate_heisenberg_circuit(num_qubits=40, time=0.1, trotter_reps=2)

        observables = [SparsePauliOp(obs) for obs in sample_observables]
        with open("obp_obs.txt", "a") as f:
            f.write(convert_observables(observables))
            f.write('\n')

    elif circuit_type == "qasm_bench":
        circuit = load_qasmbench("ising_n26", "medium")

        observables = [SparsePauliOp(obs) for obs in sample_observables]
        with open("obp_obs.txt", "a") as f:
            f.write(convert_observables(observables))
            f.write('\n')

    with open('obp_obs.txt') as f:
        content = f.readlines()
    system_size = int(content[0])

    all_observables = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        all_observables.append(one_observable)

    measurement_procedure = derandomized_classical_shadow(all_observables, int(measurements_per_observable), system_size)
    # measurement_procedure = randomized_classical_shadow(measurements_per_observable, system_size=system_size)
    if noisy:
        measurements = generate_shadow_measurements(
                                                    measurement_scheme=measurement_procedure, 
                                                    budget=10,
                                                    quantum_state_circuit=circuit,
                                                    shots_per_measurement=shots_per_measurement,
                                                    noisy=noisy
                                                )
    else:
        measurements = generate_shadow_measurements(
                                                    measurement_scheme=measurement_procedure, 
                                                    budget=10,
                                                    quantum_state_circuit=circuit,
                                                    shots_per_measurement=shots_per_measurement,
                                                    noisy=noisy
                                                )
    save_measurements_to_file(measurements, filename='measurements.txt')

    full_measurement = load_measurements('measurements.pkl')

    num_meas = len(measurements) - 1

    shadows = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        sum_product, cnt_match, products = estimate_exp(full_measurement, one_observable)
        shadows.append(sum_product / cnt_match)

    i = 0
    obs_dict = {}
    for one_obs, estimate in zip(content[1:], shadows):
        if circuit.num_qubits > 32:
            state_vector_estimator = EstimatorV2(
                options={"backend_options": {"method": "matrix_product_state"}}
            )
        else:
            state_vector_estimator = StatevectorEstimator()
        result_exact = (state_vector_estimator.run([(circuit, sample_observables[i])]).result()[0]).data.evs.item()
        obs_dict[sample_observables[i]] = (estimate, result_exact)
        i += 1

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

def main():
    """
    The observables will be generated from the OBP code
    """
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

    observables=observables=[
                            # one, two, three
                            "ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
                            "XXIIIIIIII", "XZIIIIIIII", "XYIIIIIIII",
                            "YYIIIIIIII", "YZIIIIIIII", "YXIIIIIIII",

                            # "IZZIIIIIII", "IZXIIIIIII", "IZYIIIIIII",
                            # "IXXIIIIIII", "IXZIIIIIII", "IXYIIIIIII",
                            # "IYYIIIIIII", "IYZIIIIIII", "IYXIIIIIII",

                            # "IIZZIIIIII", "IIZXIIIIII", "IIZYIIIIII",
                            # "IIXXIIIIII", "IIXZIIIIII", "IIXYIIIIII",
                            # "IIYYIIIIII", "IIYZIIIIII", "IIYXIIIIII",

                            # "IIIZZIIIII", "IIIZXIIIII", "IIIZYIIIII",
                            # "IIIXXIIIII", "IIIXZIIIII", "IIIXYIIIII",
                            # "IIIYYIIIII", "IIIYZIIIII", "IIIYXIIIII"
                            ]

    # observables = ["IIIIIIIIIIIIIZIIIIIIIIIIII",
    #                "IIIIIIIIIIIIIZZIIIIIIIIIII",
    #                "IIIIIIIIIIIIIXIIIIIIIIIIII"]
    
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data3'))
    filename = f'{data_path}/shad_noisy_res_9.pkl'

    start_meas = 1
    max_meas = 100_000

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
                    depth=10,
                    circuit_type="heisenberg_10",
                    n=10,
                    k=8,
                    noisy=True
                )
        
        meas *= 10
        
        all_data_dict_lists.append(new_data)
        meas_list.append(num_meas)

        with open('measurements.txt', 'w') as f:
            pass

        if os.path.exists('measurements.pkl'):
            os.remove('measurements.pkl')

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
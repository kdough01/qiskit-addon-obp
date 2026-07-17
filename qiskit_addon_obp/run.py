"""
This is intended to run many backpropagated circuits all at once starting from some
initial circuit and creating one measurement file with all of this data.
"""

import os
import time
import json
import pickle
import inspect
import pandas as pd
import numpy as np

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2

from qiskit_addon_obp.helpers import get_depth, save_to_pickle, get_heisenberg, get_qaoa
from qiskit_addon_obp.obp import convert_observables_for_many, obp_protocol, convert_observables_for_many
from qiskit_addon_obp.classical_shadows import derandomized_classical_shadow, estimate_exp, generate_shadow_measurements, save_measurements_to_file, generate_statevector_shadow_measurements
from qiskit_addon_obp.benchmarks.benchmark_suite import load_qasmbench

def measurement(circuit, budget, ps_pc_map, file='obp_obs.txt', measurements_per_observable=100, shots_per_measurement=100, noisy=False, use_weights=True):

    with open(file) as f:
        content = f.readlines()
    system_size = int(content[0])
    print(system_size)

    all_observables = []
    weight_params = []
    for line in content[1:]:
        one_observable = []
        obs_string = ['I' for i in range(system_size)]
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
            obs_string[int(position)] = pauli_XYZ
        if one_observable not in all_observables:
            obs = ''.join(obs_string)
            all_observables.append(one_observable)
            weight_params.append(ps_pc_map[obs])

    if use_weights:
        print(f"Beginning Derandomization with weight...")
        start = time.time()
        measurement_procedure = derandomized_classical_shadow(all_observables, int(measurements_per_observable), system_size, weight=weight_params)
    else:
        print(f"Beginning Derandomization without weights...")
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
    save_measurements_to_file(measurements, filename='measurements.pkl')

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
    
    # with open('measurements.txt') as f:
    #     measurements = f.readlines()

    with open('measurements.pkl', 'rb') as f:      # ← Changed
        full_measurement = pickle.load(f)

    # full_measurement = []
    # for line in measurements[1:]:
    #     single_meaurement = []
    #     for pauli_XYZ, outcome in zip(line.split(" ")[0::2], line.split(" ")[1::2]):
    #         single_meaurement.append((pauli_XYZ, float(outcome)))
    #     full_measurement.append(single_meaurement)

    variances = []
    shadows = []
    for line in content[1:]:
        one_observable = []
        for pauli_XYZ, position in zip(line.split(" ")[1::2], line.split(" ")[2::2]):
            one_observable.append((pauli_XYZ, int(position)))
        sum_product, cnt_match, products = estimate_exp(full_measurement, one_observable)
        if cnt_match > 1:
            var = np.var(products, ddof=1)
            shad = sum_product / cnt_match
        elif cnt_match == 1:
            var = np.nan
            shad = sum_product / cnt_match
        elif cnt_match == 0:
            var = np.nan
            shad = np.nan
        shadows.append(shad)
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
            if p_coeff and not np.isnan(obs_shad_dict[p_string]['mean']):
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

    if circuit.num_qubits > 10:
        print("using MPS")

    for idx, (obs, val) in enumerate(obs_shad_dict.items()):
        estimate = val["mean"]
        variance = val["variance"]

        if circuit.num_qubits > 10:
            state_vector_estimator = EstimatorV2(
                options={"backend_options": {"method": "matrix_product_state"}}
            )
        else:
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

def run_many(
        observables,
        circuit_type,
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
        n=5,
        k=3,
        use_weights=True
        ):
    """
    Inputs:
    - observables: list - takes in a list of observables, even if only entering one observable, must be of list form
    - which_circuit: bool - if True, the original circuit will be used, otherwise, if False, the backpropagated circuits will be used
    """

    if circuit_type == "qaoa":
        circuit, observables = get_qaoa(n, k, reps=depth)
    elif circuit_type == "heisenberg":
        circuit = get_heisenberg(depth)
        # circuit, _ = generate_heisenberg_circuit(num_qubits=40, time=0.1, trotter_reps=2)
        # circuit = get_heisenberg_40(depth)
    elif circuit_type == "qasm_bench":
        circuit = load_qasmbench("ising_n26", "medium")

    with open("obp_obs.txt", "a") as f:
        f.write(str(len(observables[0])))
        f.write('\n')

    all_data_dict_lists = []
    all_pauli_strings_list = []
    all_pauli_coeffs_list = []
    ps_list = []
    pc_list = []
    for obs in observables:
        new_data, pauli_strings_list, pauli_coeffs_list, total_shots = obp_protocol(
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
        ps_list.extend(pauli_strings_list[0])
        pc_list.extend(pauli_coeffs_list[0])

    sample_observables = []
    ps_pc_map = {}
    for ps, pc in zip(ps_list, pc_list):
        if ps in ps_pc_map:
            ps_pc_map[ps] += abs(pc.real)
        else:
            ps_pc_map[ps] = abs(pc.real)
            sample_observables.append(SparsePauliOp(ps))

    with open("obp_obs.txt", "a") as f:
        f.write(convert_observables_for_many(sample_observables))
        f.write('\n')

    if which_circuit:
        measurement(circuit=circuit,
                    budget=budget,
                    measurements_per_observable=measurements_per_observable,
                    shots_per_measurement=shots_per_measurement,
                    noisy=noisy,
                    use_weights=use_weights,
                    ps_pc_map=ps_pc_map)
    else:
        measurement(circuit=all_data_dict_lists[0]['bp_circuit'],
                    budget=budget,
                    measurements_per_observable=measurements_per_observable,
                    shots_per_measurement=shots_per_measurement,
                    noisy=noisy,
                    use_weights=use_weights,
                    ps_pc_map=ps_pc_map)

    sample_observables = reconstruct_pauli_string()

    obs_shad_dict = shadow_estimates_dict(sample_observables)

    obs_shad_df = obs_shad_exact(circuit, obs_shad_dict)
    with open('obs_shad_dict.json', 'w') as f:
        json.dump(obs_shad_df, f)

    for obs_idx in range(len(observables)):
        og_shadow_estimates(all_data_dict_lists[obs_idx], all_pauli_strings_list[obs_idx], all_pauli_coeffs_list[obs_idx], obs_shad_dict)
        all_data_dict_lists[obs_idx]['obs_shad_dict'] = obs_shad_df

    return all_data_dict_lists

def normal():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data3'))
    filename = f'{data_path}/pauli-trunc7_init90_bp5_obs9.pkl'

    data_list = []
    start = time.time()
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

    # observables = ["IIIIIIIIIIIIIZIIIIIIIIIIII",
    #                "IIIIIIIIIIIIIZZIIIIIIIIIII",
    #                "IIIIIIIIIIIIIXIIIIIIIIIIII"]

    new_data = run_many(observables=[#one, two, three
                                    "ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
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
                        circuit_type="heisenberg",
                        budget=10,
                        target_depth=10,
                        max_error_per_slice=0.0001,
                        measurements_per_observable=100,
                        shots_per_measurement=1,
                        depth=10, # this is really the number of trotter steps, so the depth is 9*depth for heis, for QAOA this sets the reps
                        which_circuit=False,
                        noisy=False,
                        obp_shots=10000,
                        coeff_truncate=False,
                        pauli_truncate=True,
                        truncation_weight=7,
                        n=10,
                        k=6,
                        use_weights=False)
    end = time.time()
    print("done with this")
    if new_data:
        with open('measurements.pkl', 'rb') as f:
            saved_measurements = pickle.load(f)
        num_measurements = len(saved_measurements)
        
        data_list += new_data

    df = pd.DataFrame(data_list)

    df['num_meas'] = num_measurements - 1
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

    if os.path.exists('measurements.pkl'):
        os.remove('measurements.pkl')

    print(f"Total Time: {end - start} seconds")

def main():
    normal()

if __name__ == '__main__':
    main()
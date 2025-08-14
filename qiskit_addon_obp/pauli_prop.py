from qiskit.transpiler import CouplingMap
import numpy as np
from juliacall import Main as jl
from qiskit_addon_obp.run import measurement, reconstruct_pauli_string, og_shadow_estimates, obs_shad_exact, shadow_estimates_dict, get_depth, save_to_pickle
from qiskit.synthesis import LieTrotter
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit
import json
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit_addon_obp.obp import convert_observables_for_many
import time
import pandas as pd
import os
from qiskit.primitives import StatevectorEstimator

# print("Julia version:", jl.VERSION)

# print("Installing PauliPropagation.jl (grab a coffee, this could take a minute)...")
# jl.seval("""
#     using Pkg
#     if !haskey(Pkg.project().dependencies, "PauliPropagation")
#         Pkg.add("PauliPropagation")
#     end
# """)

jl.seval("using PauliPropagation")
# print("Finally! PauliPropagation.jl is loaded")

def observable_to_julia(observable):

    pp = jl.PauliPropagation
    
    nqubits = len(observable)

    Xsym = jl.Symbol("X")  # For X Pauli
    Ysym = jl.Symbol("Y")  # For Y Pauli
    Zsym = jl.Symbol("Z")  # For Z Pauli


    idx_list = []
    pauli_list = []
    for idx, pauli in enumerate(list(observable)):
        if pauli == 'X':
            idx_list.append(idx+1)
            pauli_list.append(Xsym)

        if pauli == 'Y':
            idx_list.append(idx+1)
            pauli_list.append(Ysym)

        if pauli == 'Z':
            idx_list.append(idx+1)
            pauli_list.append(Zsym)

    pstr = pp.PauliString(nqubits, pauli_list, idx_list)

    return pstr

def make_xyz_pauli_circuit_julia(reps):
    # Define coupling map and reduce it to a topology
    coupling_map = CouplingMap.from_heavy_hex(3, bidirectional=False)
    topology = [0, 13, 1, 14, 10, 16, 5, 12, 8, 18]
    reduced_coupling_map = coupling_map.reduce(topology)
    edges = [(int(a), int(b)) for a, b in reduced_coupling_map.get_edges()]

    # Set Hamiltonian parameters
    jx, jy, jz = np.pi / 8, np.pi / 4, np.pi / 2
    hx, hy, hz = np.pi / 3, np.pi / 6, np.pi / 9
    time = 0.1

    # Send to Julia
    jl.edges = edges
    jl.jx, jl.jy, jl.jz = jx, jy, jz
    jl.hx, jl.hy, jl.hz = hx, hy, hz
    jl.timing = time
    jl.depth = reps

    code = """
    using PauliPropagation

    function make_xyz_pauli_circuit(edges, jx, jy, jz, hx, hy, hz, timing, depth)
        circuit = Vector{Gate}()
        nqubits = maximum(vcat(map(x -> collect(x), edges)...)) + 1

        for _ in 1:depth
            for (i, j) in edges
                # RXX gate
                push!(circuit, PauliRotation([:X, :X], [i+1, j+1], 2 * jx * timing / depth))
                # RYY gate
                push!(circuit, PauliRotation([:Y, :Y], [i+1, j+1], 2 * jy * timing / depth))
                # RZZ gate
                push!(circuit, PauliRotation([:Z, :Z], [i+1, j+1], 2 * jz * timing / depth))
            end
            for i in 0:(nqubits-1)
                push!(circuit, PauliRotation(:X, i+1, 2 * hx * timing / depth))
                push!(circuit, PauliRotation(:Y, i+1, 2 * hy * timing / depth))
                push!(circuit, PauliRotation(:Z, i+1, 2 * hz * timing / depth))
            end
        end
        return circuit
    end
    """

    jl.seval(code)

    circuit = jl.make_xyz_pauli_circuit(jl.edges, jl.jx, jl.jy, jl.jz, jl.hx, jl.hy, jl.hz, jl.timing, jl.depth)

    return circuit

def pauli_prop(observable, reps=10, bp_circuit_length=506, max_weight=7, min_abs_coeff=1e-4):
    """
    In order to perform Pauli Propagation to directly compare it to Operator Backpropagation
    you have to know the length of the circuit you want after propagation. In the tests I have been
    performing the initial circuit has a total length of 570 and the BP circuit has a length of 505
    therefore, when I do the Pauli propagation I only propagate to a length of 505 (Julia indexing starts
    at 1 so the bp_circuit_length is 506).
    """
    circuit = make_xyz_pauli_circuit_julia(reps=reps)

    obs = observable_to_julia(observable)

    nqubits = len(observable)
    jl.nqubits = nqubits

    pp = jl.PauliPropagation

    propagated_state = pp.propagate(circuit[1:bp_circuit_length], obs, max_weight=max_weight, min_abs_coeff=min_abs_coeff)

    jl.propagated_state = propagated_state
    jl.seval("""
                function pauli_label_from_bits(pauli_uint, n)
                    pauli_letters = ['I', 'X', 'Y', 'Z']
                    s = ""
                    for i = 1:n
                        shift = 2*(i-1)
                        bits = (pauli_uint >> shift) & 0x03
                        s = string(pauli_letters[bits+1], s)
                    end
                    return reverse(s)
                end
                """)

    pauli_strings_list = list(jl.seval("[pauli_label_from_bits(term.first, nqubits) for term in propagated_state]"))
    pauli_coeffs_list = list(jl.seval("[term.second for term in propagated_state]"))

    overlap = pp.overlapwithzero(propagated_state)

    new_data = {
        'obs': observable,
        'pp_exp': overlap,
        'pp_obs': SparsePauliOp(pauli_strings_list, pauli_coeffs_list),
    }

    with open("obp_obs.txt", "a") as f:
        f.write(convert_observables_for_many(new_data['pp_obs']))
        f.write('\n')

    return new_data, pauli_strings_list, pauli_coeffs_list

def run_pauli_prop(
        observables,
        bp_circuit_length=506,
        budget=4,
        measurements_per_observable=1,
        shots_per_measurement=100,
        depth=5,
        which_circuit=False,
        noisy=False,
        truncation_weight=7,
        min_abs_coeff=1e-10
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
    # all_pauli_strings_list = []
    # all_pauli_coeffs_list = []

    for obs in observables:
        new_data, pauli_strings_list, pauli_coeffs_list = pauli_prop(obs, bp_circuit_length=bp_circuit_length, max_weight=truncation_weight, min_abs_coeff=min_abs_coeff)
        state_vector_estimator = StatevectorEstimator()
        result_exact = (state_vector_estimator.run([(circuit, obs)]).result()[0]).data.evs.item()
        new_data['exact_exp'] = result_exact
        all_data_dict_lists.append(new_data)

    pauli_strings_list = []
    pauli_coeffs_list = []
    for item in all_data_dict_lists:
        sparse_pauli_op = item["pp_obs"]
        strings = [str(pauli) for pauli in sparse_pauli_op.paulis]
        pauli_strings_list.append(strings)
            
        coeffs = [complex(coeff) for coeff in sparse_pauli_op.coeffs]
        pauli_coeffs_list.append(coeffs)

    if which_circuit:
        measurement(circuit=circuit, budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)
    else:
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        for instr, qargs, cargs in circuit.data[1:506]:
            new_circuit.append(instr, qargs, cargs)

        measurement(circuit=new_circuit, budget=budget, measurements_per_observable=measurements_per_observable, shots_per_measurement=shots_per_measurement, noisy=noisy)

    sample_observables = reconstruct_pauli_string()

    obs_shad_dict = shadow_estimates_dict(sample_observables)
    obs_shad_df = obs_shad_exact(circuit, obs_shad_dict)
    with open('obs_shad_dict.json', 'w') as f:
        json.dump(obs_shad_df, f)

    for obs_idx in range(len(observables)):
        og_shadow_estimates(all_data_dict_lists[obs_idx], [pauli_strings_list[obs_idx]], [pauli_coeffs_list[obs_idx]], obs_shad_dict)
        # all_data_dict_lists[obs_idx]['obs_shad_dict'] = obs_shad_df

    return all_data_dict_lists

def main():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'pauli_prop_data'))
    filename = f'{data_path}/pp-shad-pauli4_coef4_init90_bp70_obs3.pkl'

    data_list = []
    start = time.time()
    new_data = run_pauli_prop(observables=["ZZIIIIIIII", "ZXIIIIIIII", "ZYIIIIIIII",
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
                        bp_circuit_length=506, # 506 gives us a depth of 70
                        measurements_per_observable=100,
                        shots_per_measurement=1,
                        depth=10,
                        which_circuit=False,
                        noisy=False,
                        truncation_weight=4,
                        min_abs_coeff=1e-4
                        )
    end = time.time()
    
    if new_data:
        with open('measurements.txt', 'r') as f:
            file = f.readlines()
        data_list += new_data

    df = pd.DataFrame(data_list)

    df['num_meas'] = len(file) - 1

    df['abs_pp_error'] = abs(df['exact_exp'] - df['pp_exp'])
    df['abs_total_error'] = abs(df['exact_exp'] - df['total_exp'])

    df['pp_error'] = df['exact_exp'] - df['pp_exp']
    df['total_error'] = df['exact_exp'] - df['total_exp']

    save_to_pickle(df, filename)

    with open("measurements.txt", "rb") as f:
        file_size = os.path.getsize('measurements.txt')
        print(f"Measurement file size: {file_size} bytes")

    with open("obp_obs.txt", "rb") as f:
        file_size = os.path.getsize('obp_obs.txt')
        print(f"Observable file size: {file_size} bytes")

    with open('measurements.txt', 'w') as f:
        pass

    with open('obp_obs.txt', 'w') as f:
        pass

    print(f"Total Time: {end - start} seconds")
if __name__ == "__main__":
    main()
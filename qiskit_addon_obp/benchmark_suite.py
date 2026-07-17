from qiskit import QuantumCircuit
import os

def load_qasmbench(benchmark_name: str, size: str = "small", remove_measurements=True):
    """
    Load any circuit from QASMBench/small
    """
    base = "/Users/kevindougherty/Documents/GitHub/QASMBench"   # ← Change if your path is different
    
    # Most common naming patterns in QASMBench
    possible_paths = [
        f"{base}/{size}/{benchmark_name}/{benchmark_name}.qasm",
        f"{base}/{size}/{benchmark_name}_{size}/{benchmark_name}.qasm",
        f"{base}/{size}/{benchmark_name}/circuit.qasm",
        f"{base}/{size}/{benchmark_name}/{benchmark_name}/circuit.qasm",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            qc = QuantumCircuit.from_qasm_file(path)
            if remove_measurements:
                qc = qc.remove_final_measurements(inplace=False)
            print(f"Loaded: {benchmark_name} | Qubits: {qc.num_qubits} | Depth: {qc.depth()}")
            return qc
    
    print(f"Could not find {benchmark_name}")
    return None

if __name__ == "__main__":
    qc = load_qasmbench("qft_n4", "small")
    print(qc)
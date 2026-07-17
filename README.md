# Modified Qiskit addon: operator backpropagation (OBP)

This repository is a research fork of the Qiskit Addon for Operator Backpropagation (OBP).

It extends the original implementation with:
- Depth-based truncation
- Additional truncation strategies
- Classical shadows integration

## New Features in This Fork

### Depth-Based Truncation
We now allow a user to specify a target depth that does not depend on qubit-wise commuting (QWC) groups. This may result in larger observables being generated, but multiple original observables can now run on the same backpropagated circuit. This is useful for integration with classical shadows.

----------------------------------------------------------------------------------------------------

### Additional Truncation Strategies
Pauli weight truncation - we added the ability to truncate based on Pauli weight

Hybrid truncation - we allow users to truncate with both strategies (coefficient and Pauli-weight). We choose to first truncate by small coefficients up to some predetermined slice error. We then take the resulting observable and further truncate any high weight Paulis. However, it is known that many high-weight Pauli observables will inherently have small coefficient terms. Therefore, it makes more sense to first truncate by small coefficients up to a slice error where many of the terms have large Pauli-weights, and then remove any remaining large Pauli-weight terms after. Note that high-weight Pauli terms that are leftover after truncating by low coefficient weight do not contribute to the error allocation.

----------------------------------------------------------------------------------------------------

### Classical Shadows Integration
Copying the functions from https://github.com/hsinyuan-huang/predicting-quantum-properties?tab=readme-ov-file, we integrate backpropagation with their classical shadows protocol. Ignoring the size of the backpropagated observable, and backpropagating to a specific depth was important here if we wanted to be able to backpropagate multiple observables and have each backpropagated observable used in the formation of the shadow.

`run.py` will run both the backpropagation and classical shadows and save the data to a pickle file. It will calculate and return the expected value of the original observable, as well as the error incurred. There are a few functions that will run the entire protocol that were used for testing and exploration, but the function to generate the data in the analysis notebooks is simply called `normal`. Multiple observables may be specified, along with many other flags, though most tests will want to be run with the given presets.

`classical_shadows.py` will run the full classical shadows protocol on the initial circuit and save the expected values to a pickle file. Multiple observables may be specified. You may also specify a number of measurements the protocol will run. The loop automatically resets the measurements each time, so simply setting an initial number of measurements per observable and a maximum measurements per observable will suffice.

`obp.py` will run operator backpropagation generally following the steps provided on Qiskit's tutorial page.

----------------------------------------------------------------------------------------------------

### Sources
This repository is an adaptation of the original Operator Backpropagation Qiskit Addon, and the original README is displayed below. A link to that repository can be found here:
- https://github.com/Qiskit/qiskit-addon-obp

This repository also copies functions from the Predicting Properties of Quantum Many-Body Systems repository, linked here:
- https://github.com/hsinyuan-huang/predicting-quantum-properties?tab=readme-ov-file
The code from this repository exists in the `classical_shadows.py` file and the functions are specified in their docstrings.

The additional truncation method was inspired by the papers associated with the Pauli Propagation repository, which was linked above, but can be found here:
- https://github.com/MSRudolph/PauliPropagation.jl?tab=readme-ov-file


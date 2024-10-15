import numpy as np

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms.eigensolvers import NumPyEigensolver

from fromhopetoheuristics.utils.qaoa_utils import hamiltonian_from_qubo


def build_mixing_hamiltonian(num_qubits):
    # As in
    # https://qiskit.org/documentation/_modules/qiskit/circuit/library/
    # n_local/qaoa_ansatz.html#QAOAAnsatz
    mixer_terms = [
        ("I" * left + "X" + "I" * (num_qubits - left - 1), 1)
        for left in range(num_qubits)
    ]
    mixer_hamiltonian = SparsePauliOp.from_list(mixer_terms)
    return mixer_hamiltonian


# Builds the interpolation hamiltonian for the given problem hamiltonian,
# fraction s and amount of qubits
def build_hamiltonian(qubo, fraction, num_qubits):
    cost_hamiltonian, offset = hamiltonian_from_qubo(qubo)
    mixer_hamiltonian = build_mixing_hamiltonian(num_qubits)
    offset_hamiltonian = SparsePauliOp(["I"])
    for i in range(num_qubits - 1):
        offset_hamiltonian = offset_hamiltonian.tensor(Pauli("I"))
    offset_hamiltonian = offset_hamiltonian._multiply(offset)
    cost_hamiltonian = cost_hamiltonian._add(offset_hamiltonian)
    mixer_hamiltonian = mixer_hamiltonian._multiply((1 - fraction))
    cost_hamiltonian = cost_hamiltonian._multiply(fraction)
    H = mixer_hamiltonian._add(cost_hamiltonian)
    return H


# Derives the spectral gap for the given problem hamiltonian at the given fraction s
def calculate_spectral_gap(fraction: float, qubo: np.ndarray, num_dec_pos: int = 4):
    num_qubits = len(qubo)
    H = build_hamiltonian(qubo, fraction, num_qubits)
    counter = 0
    eigenvalues = []
    # eigenstates = []
    while len(set(eigenvalues)) < 2 and counter != num_qubits:
        # Increase the counter in every iteration, such that the number
        # of searched eigenvalues exponentially increases
        # as long as no two unique eigenvalues are found
        counter = counter + 1
        eigensolver = NumPyEigensolver(k=pow(2, counter))
        eigensolver_result = eigensolver.compute_eigenvalues(H)
        # eigenstates = eigensolver_result.eigenstates
        eigenvalues = np.real(eigensolver_result.eigenvalues)
        eigenvalues = np.around(eigenvalues, num_dec_pos)

    eigenvalues = np.real(np.unique(eigenvalues))
    spectral_gap = np.around(np.abs(eigenvalues[0] - eigenvalues[1]), num_dec_pos)
    return eigenvalues[0], eigenvalues[1], spectral_gap

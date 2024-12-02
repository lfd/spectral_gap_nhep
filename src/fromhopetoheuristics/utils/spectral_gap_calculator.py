import numpy as np
from typing import List, Iterable, Tuple, Dict, Union

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms.eigensolvers import NumPyEigensolver

from fromhopetoheuristics.utils.qaoa_utils import hamiltonian_from_qubo


def build_mixing_hamiltonian(num_qubits: int) -> SparsePauliOp:
    """
    Builds a mixing hamiltonian for the given number of qubits.
    This is built according to the QAOAAnsatz implementation in Qiskit.
    See https://qiskit.org/documentation/_modules/qiskit/circuit/library/
    n_local/qaoa_ansatz.html#QAOAAnsatz

    Args:
        num_qubits: The number of qubits in the hamiltonian to build

    Returns:
        A SparsePauliOp representing the mixing hamiltonian
    """
    mixer_terms = [
        ("I" * left + "X" + "I" * (num_qubits - left - 1), 1)
        for left in range(num_qubits)
    ]
    mixer_hamiltonian = SparsePauliOp.from_list(mixer_terms)
    return mixer_hamiltonian


def build_hamiltonian(
    qubo: np.ndarray, fraction: float, num_qubits: int
) -> SparsePauliOp:
    """
    Builds the interpolation hamiltonian for the given problem hamiltonian,
    fraction s and amount of qubits

    Args:
        qubo: The QUBO matrix
        fraction: The interpolation fraction s, s in [0,1]
        num_qubits: The number of qubits in the hamiltonian to build

    Returns:
        A SparsePauliOp representing the interpolation hamiltonian
    """
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


def calculate_spectral_gap(
    fraction: float, qubo: np.ndarray, num_dec_pos: int = 4
) -> Tuple[float, float, float]:
    """
    Calculates the spectral gap for a given problem hamiltonian at a given fraction s.

    Args:
        fraction: The interpolation fraction s, s in [0,1]
        qubo: The QUBO matrix
        num_dec_pos: The number of decimal positions to round the eigenvalues to

    Returns:
        A tuple of three floats, the ground state energy,
        the first excited state energy, and the spectral gap
    """
    num_qubits = len(qubo)
    H = build_hamiltonian(qubo, fraction, num_qubits)
    counter = 0
    eigenvalues: List[float] = []
    # eigenstates: List[np.ndarray] = []
    while len(set(eigenvalues)) < 2 and counter != num_qubits:
        # Increase the counter in every iteration, such that the number
        # of searched eigenvalues exponentially increases
        # as long as no two unique eigenvalues are found
        counter += 1
        eigensolver = NumPyEigensolver(k=pow(2, counter))
        eigensolver_result = eigensolver.compute_eigenvalues(H)
        # eigenstates = eigensolver_result.eigenstates
        eigenvalues = np.real(eigensolver_result.eigenvalues)
        eigenvalues = np.around(eigenvalues, num_dec_pos)

    eigenvalues = np.real(np.unique(eigenvalues))
    spectral_gap = np.around(np.abs(eigenvalues[0] - eigenvalues[1]), num_dec_pos)
    return eigenvalues[0], eigenvalues[1], spectral_gap


def annealing(
    qubo: np.ndarray,
    fractions: Iterable[float],
) -> List[Dict[str, Union[float, Tuple[float, float, float]]]]:
    """
    Calculates the spectral gap for a given problem hamiltonian at
    a given interpolation fraction s.

    Args:
        qubo: The QUBO matrix
        fractions: The interpolation fractions s, s in [0,1]

    Returns:
        A list of dictionaries, where each dictionary contains the
        fraction, ground state energy, first excited state energy,
        and the spectral gap
    """
    results: List[Dict[str, Union[float, Tuple[float, float, float]]]] = []

    for fraction in fractions:
        res = {"fraction": fraction}
        res["gs"], res["fes"], res["gap"] = calculate_spectral_gap(fraction, qubo)
        results.append(res)

    return results

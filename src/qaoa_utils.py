import numpy as np
from typing import Dict, Tuple, Optional
from qiskit.quantum_info import SparsePauliOp

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()


def hamiltonian_from_qubo(qubo: np.ndarray) -> Tuple[SparsePauliOp, float]:
    """
    Creates Qiskit Hamiltonian based on the QUBO matrix

    :param J: np.ndarray: Ising matrix
    :param h: np.ndarray: linear Ising terms
    :param o: np.ndarray: offset

    :return: SparsePauliOp: Hamiltonian
    :return: float: Offset
    """
    J, h, o = pure_QUBO_to_ising(qubo)
    return hamiltonian_from_ising(J, h, o)


def hamiltonian_from_ising(
    J: np.ndarray, h: np.ndarray, o: float
) -> Tuple[SparsePauliOp, float]:
    """
    Creates Qiskit Hamiltonian based on the Ising matrix and linear terms

    :param J: np.ndarray: Ising matrix
    :param h: np.ndarray: linear Ising terms
    :param o: np.ndarray: offset

    :return: SparsePauliOp: Hamiltonian
    :return: float: Offset
    """

    # Initialize empty Hamiltonian
    terms = []

    # Linear Terms
    if h is not None:
        term = ["I"] * len(J)
        for i, angle in enumerate(h):
            if angle > 0:
                term[i] = "Z"
                t = "".join(term)
                terms.append((t, angle))

    # Quadratic Terms (Assuming a Ising matrix with zero diagonal elements)
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            term = ["I"] * len(J)
            if J[i][j] > 0 or J[j][i] > 0:
                term[i] = "Z"
                term[j] = "Z"
                t = "".join(term)
                terms.append((t, J[i][j] + J[j][i]))

    hamiltonian = SparsePauliOp.from_list(terms)

    return hamiltonian, o


def pure_ising_to_QUBO(J: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate QUBO Matrix Q and offset E0 from J, such that
    s^T J s equals x^T Q x + E0
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    :param J: np.ndarray: Ising Matrix

    :return: np.ndarray: QUBO Matrix
    :return: float: Offset
    """
    n = J.shape[0]
    qubo = 4 * (J - np.diag(np.ones(n).T @ J))
    return qubo, np.sum(J)


def pure_QUBO_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate Ising Matrix J and offset o from Q, such that
    s^T J s + o equals x^T Q x
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    :param Q: np.ndarray: QUBO Matrix

    :return: np.ndarray: Quadratic Ising matrix
    :return: np.ndarray: Linear terms
    :return: float: Offset
    """
    J = 0.25 * Q
    np.fill_diagonal(J, 0.0)

    h = 0.5 * np.sum(Q, axis=1)
    o = 0.25 * (np.sum(Q) + np.trace(Q))
    return J, h, o


def provide_random_QUBO(nqubits: int, problem_seed: int = 777) -> np.ndarray:
    """
    Generates a randomly created QUBO from uniform distribution

    :param nqubits: int: Number of qubits / nodes in the problem
    :param problem_seed: Seed for numpys default random number generator

    :return: np.ndarray: QUBO matrix
    """
    global problems

    prob_key = f"random_{nqubits}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    a = np.random.default_rng(seed=problem_seed).random((nqubits, nqubits))
    qubo = np.tril(a) + np.tril(a, -1).T
    qubo = (qubo - 0.5) * 2
    problems[prob_key] = qubo
    return qubo


def cost_function(
    J: np.ndarray,
    x: np.ndarray,
    h: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes x^T * J * x + hT * x + o for a given batch

    :param J: np.ndarray: QUBO or Ising matrix
    :param x: np.ndarray: Variable assignment either in [0,1] (QUBO)
              or in [-1, 1] (Ising)
    :param h: Optional(np.ndarray): Linear terms for Ising model (Default None)
    :param offset: Optional(np.ndarray): Ising model offset (Default 0)

    :return: np.ndarray: Batched costs for variable assignments
    """
    Jx = np.einsum("bij,bj->bi", J, x)
    cost = np.einsum("bi,bi->b", x, Jx)

    if h is not None:
        cost += np.einsum("bi,bi->b", x, h)

    if offset is not None:
        cost += offset

    return cost


def dict_QUBO_to_matrix(dict_qubo: Dict[Tuple[str, str], float]) -> np.ndarray:
    """
    Transforms QUBO in dictinary form to matrix

    :param dict_qubo: QUBO dict
    :type dict_qubo: Dict[Tuple[str, str], float]
    :return: QUBO matrix
    :rtype: np.ndarray
    """
    # get unique set of variable names
    names = []
    for k in dict_qubo.keys():
        names.append(k[0])
        names.append(k[1])
    names = list(set(names))
    n_vars = len(names)

    qubo_matrix = np.zeros((n_vars, n_vars))

    for k, v in dict_qubo.items():
        qubo_matrix[names.index(k[0])][names.index(k[1])] = v

    return qubo_matrix

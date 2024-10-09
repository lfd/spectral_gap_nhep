import networkx as nx
import numpy as np
from typing import Tuple, Optional
from qiskit.quantum_info import SparsePauliOp

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()

def hamiltonian_from_qubo(qubo: np.ndarray) -> Tuple[any, float]:
    J, h, o = pure_QUBO_to_ising(qubo)
    return hamiltonian_from_ising(J, h, o)

def hamiltonian_from_ising(J: np.ndarray, h: np.ndarray, o: float) -> Tuple[any, float]:
    """
    Creates qiskit hamiltonian based on the Ising matrix and linear terms

    :param J: np.ndarray: Ising matrix
    :param h: np.ndarray: linear Ising terms
    :param o: np.ndarray: offset

    :return: tbd
    """

    # Initialize empty Hamiltonian
    coefficients, op_list = [], []
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
        for j in range(i+1, n):
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
    Calculate Qubo Matrix Q and offset E0 from J, such that
    s^T J s equals x^T Q x + E0
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    :param J: np.ndarray: Ising Matrix

    :return: np.ndarray: QUBO Matrix
    :return: float: Offset
    """
    n = J.shape[0]
    qubo = 4*(J - np.diag(np.ones(n).T @ J))
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
    J = 0.25*Q
    np.fill_diagonal(J, 0.)

    h = 0.5 * np.sum(Q, axis=1)
    o = 0.25 * (np.sum(Q) + np.trace(Q))
    return J, h, o


def maxcut_graph_to_ising(G: nx.Graph) -> Tuple[np.ndarray, float]:
    """
    Calculates Ising model from MAXCUT graph

    :param G: nx.Graph: MAXCUT graph

    :return: np.ndarray: Ising matrix
    :return: Ising offset
    """
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    m_ising = 0.25 * adjacency_matrix
    offset = - 0.25 * np.sum(adjacency_matrix)

    return m_ising, offset


def maxcut_graph_to_qubo(G: nx.Graph) -> np.ndarray:
    """
    Calculates QUBO matrix from MAXCUT graph

    :param G: nx.Graph: MAXCUT graph

    :return: np.ndarray: QUBO matrix
    """

    adjacency_matrix = nx.adjacency_matrix(G).todense()
    n = adjacency_matrix.shape[0]

    qubo = adjacency_matrix - np.diag(np.ones(n).T @ adjacency_matrix)

    return qubo


def provide_random_QUBO(nqubits: int, problem_seed: int = 777) \
        -> np.ndarray:
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
    qubo = (qubo-0.5)*2
    problems[prob_key] = qubo
    return qubo


def provide_random_maxcut_ising(nqubits: int, p: float, problem_seed: int = 777) \
        -> Tuple[np.ndarray, float]:
    """
    Generates random MaxCut Instances from Erdos-Renyi-Graphs
    The resulting graph gets mapped to a Ising matrix and offset.

    :param nqubits: int: Number of nodes (and number of qubits)
    :param p: float: Probability of randomly added edges
    :param problem_seed: int=777: Random seed for networkx graph creation

    :return: np.ndarray: Quadratic Ising matrix
    :return: float: Offset
    """
    global problems

    prob_key = f"ising_MC_{nqubits}_{p}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    g = nx.generators.erdos_renyi_graph(nqubits, p, seed=problem_seed)
    m_ising, offset = maxcut_graph_to_ising(g)

    problems[prob_key] = (m_ising, offset)
    return m_ising, offset


def provide_random_maxcut_QUBO(nqubits: int, p: float, problem_seed: int = 777) \
        -> np.ndarray:
    """
    Generates random MaxCut Instances from Erdos-Renyi-Graphs
    The resulting graph gets mapped to a QUBO.

    :param nqubits: int: Number of nodes (and number of qubits)
    :param p: float: Probability of randomly added edges
    :param problem_seed: int=777: Random seed for networkx graph creation

    :return: np.ndarray: QUBO Matrix
    """
    global problems

    prob_key = f"qubo_MC_{nqubits}_{p}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    g = nx.generators.erdos_renyi_graph(nqubits, p, seed=problem_seed)
    qubo = maxcut_graph_to_qubo(g)

    problems[prob_key] = qubo
    return qubo

def cost_function(J: np.ndarray,
                  x: np.ndarray,
                  h: Optional[np.ndarray] = None,
                  offset: Optional[np.ndarray] = None,
                  ) \
        -> np.ndarray:
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


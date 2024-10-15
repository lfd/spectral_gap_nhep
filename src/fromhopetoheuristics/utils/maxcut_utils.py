import networkx as nx
import numpy as np
from typing import Tuple

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()


def maxcut_graph_to_ising(G: nx.Graph) -> Tuple[np.ndarray, float]:
    """
    Calculates Ising model from MAXCUT graph

    :param G: nx.Graph: MAXCUT graph

    :return: np.ndarray: Ising matrix
    :return: Ising offset
    """
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    m_ising = 0.25 * adjacency_matrix
    offset = -0.25 * np.sum(adjacency_matrix)

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


def provide_random_maxcut_ising(
    nqubits: int, p: float, problem_seed: int = 777
) -> Tuple[np.ndarray, float]:
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


def provide_random_maxcut_QUBO(
    nqubits: int, p: float, problem_seed: int = 777
) -> np.ndarray:
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

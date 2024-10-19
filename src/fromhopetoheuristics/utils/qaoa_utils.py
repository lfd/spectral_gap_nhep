import numpy as np
from typing import Dict, Tuple, Optional, List
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from scipy.optimize import minimize

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()


def hamiltonian_from_qubo(qubo: np.ndarray) -> Tuple[SparsePauliOp, float]:
    """
    Creates Qiskit Hamiltonian based on the QUBO matrix

    :param qubo: np.ndarray: QUBO matrix

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
    names.sort()
    n_vars = len(names)

    qubo_matrix = np.zeros((n_vars, n_vars))

    for k, v in dict_qubo.items():
        qubo_matrix[names.index(k[0])][names.index(k[1])] = v

    return qubo_matrix


def compute_min_energy_solution(qubo: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the minimum energy solution for a given QUBO

    :param qubo: QUBO to be solved excact
    :type qubo: np.ndarray
    :return: Energy of optimal solution
    :rtype: float
    :return: Optimal solution in form of the variable assignment
    :rtype: np.ndarray
    """
    H, o = hamiltonian_from_qubo(qubo)

    eigensolver = NumPyEigensolver()
    eigensolver_result = eigensolver.compute_eigenvalues(H)

    min_eigenvalue = np.real(eigensolver_result.eigenvalues[0])
    eigenvector = eigensolver_result.eigenstates[0]
    sv = Statevector(eigenvector)
    solution = sv.sample_counts(1).popitem()[0]

    min_energy = min_eigenvalue + o

    return min_energy, solution


def initialise_QAOA_parameters(
    p: int,
    random_init: bool = False,
    seed: int = 12345,
    initial_params: Optional[np.ndarray] = None,
    fourier: bool = False,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Constructs a list of initialisation parameters for QAOA

    :param p: Number of QAOA layers
    :type p: int
    :param random_init: Whether to initialise randomly, defaults to False
    :type random_init: bool, optional
    :param seed: random_seed, defaults to 12345
    :type seed: int, optional
    :param initial_params: previous parameter initialisations that should be
        re-used, defaults to None
    :type initial_params: Optional[np.ndarray], optional
    :param fourier: if fourier strategy is used
    :type fourier: bool, optional
    :return: List of initial parameters
    :rtype: List[float]
    :return: Bounds of the parameter space
    :rtype: List[Tuple[float, float]]
    """
    if initial_params is not None:
        n_prev = len(initial_params) // 2
        prev_betas = initial_params[:n_prev]
        prev_gammas = initial_params[n_prev:]
        remaining_betas = np.repeat(0.0, p - n_prev)
        remaining_gammas = np.repeat(0.0, p - n_prev)
        beta_init = np.concatenate([prev_betas, remaining_betas])
        gamma_init = np.concatenate([prev_gammas, remaining_gammas])
    elif random_init:
        rng = np.random.default_rng(seed=seed)
        beta_init = rng.random(p) * np.pi - np.pi * 0.5
        gamma_init = rng.random(p) * 2 * np.pi - np.pi
    elif fourier:
        beta_init = np.zeros(p)
        gamma_init = np.zeros(p)
    else:
        beta_init = np.repeat(0.5 * np.pi, p)
        gamma_init = np.zeros(p)

    init_params = np.concatenate([beta_init, gamma_init])

    if fourier:
        if p < 5:
            bounds = [(-1/p, 1/p)] * 2 * p
        else:
            bounds = [(-0.25, 0.25)] * 2 * p
    else:
        bounds_beta = (-0.5 * np.pi, 0.5 * np.pi)
        bounds_gamma = (-np.pi, np.pi)
        bounds = [bounds_beta] * p + [bounds_gamma] * p

    return init_params, bounds


def get_FOURIER_params(
    v_params: np.ndarray, u_params: np.ndarray, p: int, q: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    if q == -1:
        assert len(v_params) == len(u_params) == p, (
            "Length of the parameter vector without FOURIER stragety should",
            f"be {p}, got {len(v_params)} and {len(u_params)}",
        )
        return v_params, u_params
    else:
        assert len(u_params) == len(v_params) == q, (
            f"Length of parameter vector with FOURIER stragety should be {q}",
            f"got {len(v_params)} and {len(u_params)}",
        )
        betas = np.zeros(p)
        gammas = np.zeros(p)

        for i in range(1, p + 1):
            betas[i - 1] = sum(
                [
                    v_params[k - 1] * np.cos((k - 0.5) * (i - 0.5) * np.pi / p)
                    for k in range(1, q + 1)
                ]
            )
            gammas[i - 1] = sum(
                [
                    u_params[k - 1] * np.sin((k - 0.5) * (i - 0.5) * np.pi / p)
                    for k in range(1, q + 1)
                ]
            )
        return betas, gammas


def solve_QUBO_with_QAOA(
    qubo: np.ndarray,
    p: int,
    q: int = -1,
    seed: int = 12345,
    random_param_init: bool = False,
    initial_params: Optional[np.ndarray] = None,
    optimiser: str = "COBYLA",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, o = hamiltonian_from_qubo(qubo)
    circ = QAOAAnsatz(cost_operator=H, reps=p)
    estimator = Estimator(seed=seed)
    if q == -1:
        init_params, bounds = initialise_QAOA_parameters(
            p, random_param_init, seed, initial_params, fourier=False
        )
    else:
        init_params, bounds = initialise_QAOA_parameters(
            q, random_param_init, seed, initial_params, fourier=True
        )

    def cost_fkt(
        params: np.ndarray,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: Estimator,
        p: int,
        q: int = -1,
    ) -> float:

        if q != -1:
            v_params, u_params = (
                params[: len(params) // 2],
                params[len(params) // 2 :],
            )
            betas, gammas = get_FOURIER_params(v_params, u_params, p, q)
            qaoa_params = np.concatenate([betas, gammas])
        else:
            qaoa_params = params

        qaoa_circ = ansatz.assign_parameters(qaoa_params)
        job = estimator.run([(qaoa_circ, hamiltonian)])
        result = job.result()[0]
        cost = result.data.evs
        return cost

    min_result = minimize(
        cost_fkt,
        init_params,
        args=(circ, H, estimator, p, q),
        method=optimiser,
        tol=1e-3,
        bounds=bounds,
    )
    if q == -1:
        v_params, u_params = np.array(()), np.array(())
        betas, gammas = min_result.x[:p], min_result.x[p:]
    else:
        v_params, u_params = min_result.x[:q], min_result.x[q:]
        betas, gammas = get_FOURIER_params(v_params, u_params, p, q)

    energy = min_result.fun + o
    return energy, betas, gammas, u_params, v_params


def times_from_QAOA_params(betas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    Derive midpoints of each time_interval (gamma_i + beta_i)

    :param betas: Beta parameters
    :type betas: np.ndarray
    :param gammas: Gamma parameters
    :type gammas: np.ndarray
    :return: Array of times
    :rtype: np.ndarray
    """
    p = len(betas)
    time = 0.0
    time_midpoints = np.zeros(p + 1)
    for i in range(p):
        time_segment = np.abs(gammas[i]) + np.abs(betas[i])
        time += time_segment
        time_midpoints[i] = time - 0.5 * time_segment

    time_midpoints[p] = time  # Add total annealing time
    return time_midpoints


def annealing_schedule_from_QAOA_params(
    betas: np.ndarray, gammas: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Derive Annealing schedule from QAOA parameters according to Zhou et al.
    (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.021067, Sec. VB)

    :param betas: Beta parameters
    :type betas: np.ndarray
    :param gammas: Gamma parameters
    :type gammas: np.ndarray
    :return: Annealing schedule (e.g. for Dwave), contains a list of tuples of
        the form (anneal_time, anneal_fraction).
    :rtype: List[Tuple[float, float]]
    """
    p = len(betas)
    times = times_from_QAOA_params(betas, gammas)
    anneal_schedule = [(0.0, 0.0)]
    last_added_time = 0.0
    for i in range(p):
        # ensure that time is increasing monotonically, if not, skip
        if times[i] < last_added_time:
            continue
        anneal_fraction = np.abs(gammas[i]) / (
            np.abs(gammas[i]) + np.abs(betas[i])
        )
        anneal_schedule.append((times[i], anneal_fraction))
        last_added_time = times[i]

    # at the end of the anneal schedule, the annealing fraction is 1
    anneal_schedule.append((times[p], 1.0))
    return anneal_schedule

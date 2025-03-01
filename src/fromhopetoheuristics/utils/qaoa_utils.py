import numpy as np
import logging
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds, OptimizeResult
from typing import Dict, Tuple, Optional, List, Any

log = logging.getLogger(__name__)

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()


def hamiltonian_from_qubo(qubo: np.ndarray) -> Tuple[SparsePauliOp, float]:
    """
    Creates Qiskit Hamiltonian based on the QUBO matrix

    Args:
        qubo (np.ndarray): QUBO matrix

    Returns:
        Tuple[SparsePauliOp, float]: (Hamiltonian, Offset)
    """
    J, h, o = pure_QUBO_to_ising(qubo)
    return hamiltonian_from_ising(J, h, o)


def hamiltonian_from_ising(
    J: np.ndarray, h: Optional[np.ndarray], o: float
) -> Tuple[SparsePauliOp, float]:
    """
    Creates a Qiskit Hamiltonian based on the Ising matrix and linear terms.

    :param J: np.ndarray: The Ising matrix representing quadratic terms.
    :param h: Optional[np.ndarray]: Linear Ising terms, can be None.
    :param o: float: Offset value.

    :return: Tuple[SparsePauliOp, float]: A tuple containing the Hamiltonian
        as a SparsePauliOp and the offset.
    """
    # Initialize an empty list to store Hamiltonian terms
    terms: List[Tuple[str, float]] = []

    # Add linear terms to the Hamiltonian
    if h is not None:
        for i, angle in enumerate(h):
            if angle != 0:
                term: List[str] = ["I"] * len(J)
                term[i] = "Z"
                t = "".join(term)
                terms.append((t, angle))

    # Add quadratic terms to the Hamiltonian (assuming Ising matrix
    # with zero diagonal elements)
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            term = ["I"] * len(J)
            if J[i][j] > 0 or J[j][i] > 0:
                term[i] = "Z"
                term[j] = "Z"
                t = "".join(term)
                terms.append((t, J[i][j] + J[j][i]))

    # Create a SparsePauliOp from the list of terms
    hamiltonian: SparsePauliOp = SparsePauliOp.from_list(terms)

    return hamiltonian, o


def pure_ising_to_QUBO(
    J: np.ndarray,  # Ising Matrix
) -> Tuple[np.ndarray, float]:  # QUBO Matrix and Offset
    """
    Calculate QUBO Matrix Q and offset E0 from J, such that
    s^T J s equals x^T Q x + E0
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    Args:
        J (np.ndarray): Ising Matrix

    Returns:
        Tuple[np.ndarray, float]: QUBO Matrix and Offset
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

    :return: Tuple[np.ndarray, np.ndarray, float]:
            - Quadratic Ising matrix
            - Linear terms
            - Offset
    """
    J: np.ndarray = 0.25 * Q
    np.fill_diagonal(J, 0.0)

    h: np.ndarray = 0.5 * np.sum(Q, axis=1)
    o: float = 0.25 * (np.sum(Q) + np.trace(Q))
    return J, h, o


def provide_random_QUBO(nqubits: int, problem_seed: int = 777) -> np.ndarray:
    """
    Generates a randomly created QUBO from uniform distribution.

    Args:
        nqubits (int): Number of qubits / nodes in the problem.
        problem_seed (int, optional): Seed for numpy's default random number generator.

    Returns:
        np.ndarray: QUBO matrix.
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
    J: np.ndarray,  # QUBO or Ising matrix
    x: np.ndarray,  # Variable assignment either in [0,1] (QUBO) or in [-1, 1] (Ising)
    h: Optional[np.ndarray] = None,  # Linear terms for Ising model (Default None)
    offset: Optional[float] = 0.0,  # Ising model offset (Default 0)
) -> np.ndarray:  # Batched costs for variable assignments
    """
    Computes x^T * J * x + hT * x + o for a given batch

    :param J: np.ndarray: QUBO or Ising matrix
    :param x: np.ndarray: Variable assignment either in [0,1] (QUBO)
              or in [-1, 1] (Ising)
    :param h: Optional[np.ndarray]: Linear terms for Ising model (Default None)
    :param offset: Optional[float]: Ising model offset (Default 0)

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
    Transforms QUBO in dictionary form to matrix

    Args:
        dict_qubo (Dict[Tuple[str, str], float]): QUBO dictionary

    Returns:
        np.ndarray: QUBO matrix
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
        qubo_matrix[names.index(k[0])][names.index(k[1])] = 0.5 * v
        qubo_matrix[names.index(k[1])][names.index(k[0])] = 0.5 * v

    return qubo_matrix


def compute_min_energy_solution(qubo: np.ndarray) -> Tuple[float, str]:
    """
    Computes the minimum energy solution for a given QUBO

    :param qubo: QUBO to be solved exactly
    :type qubo: np.ndarray
    :return: The energy of the optimal solution
    :rtype: float
    :return: The optimal solution in form of the variable assignment
    :rtype: str
    """
    H, o = hamiltonian_from_qubo(qubo)

    eigensolver = NumPyEigensolver()
    eigensolver_result = eigensolver.compute_eigenvalues(H)

    min_eigenvalue: float = np.real(eigensolver_result.eigenvalues[0])
    eigenvector = eigensolver_result.eigenstates[0]
    sv = Statevector(eigenvector)
    solution: str = sv.sample_counts(1).popitem()[0]

    min_energy: float = min_eigenvalue + o

    return min_energy, solution


def initialise_QAOA_parameters(
    p: int,
    initialisation: str,
    rng: np.random.Generator,
    initial_params: Optional[np.ndarray] = None,
    fourier: bool = False,
    n_pert: int = 0,
) -> np.ndarray:
    """
    Constructs a list of initialisation parameters for QAOA

    Parameters
    ----------
    p : int
        Number of QAOA layers
    initialisation : str
        Initialisation strategy for QAOA parameters.
    rng : np.random.Generator,
        The random number generator for random initialisation.
    initial_params : Optional[np.ndarray], optional
        Previous parameter initialisations that should be re-used. Defaults to
        None.
    fourier : bool, optional
        Whether fourier strategy is used. Defaults to False.
    n_pert : int, optional
        The number of random perturbations for the FOURIER strategy. Defaults
        to 0.

    Returns
    -------
    np.ndarray
        List of initial parameters
    """
    if initial_params is None or "all" in initialisation:
        n_remaining = p
        prev_betas, prev_gammas = np.tile([], (n_pert + 1, 1)), np.tile(
            [], (n_pert + 1, 1)
        )
    else:
        n_prev = len(initial_params) // 2
        n_remaining = p - n_prev
        prev_betas = np.tile(initial_params[:n_prev], (n_pert + 1, 1))
        prev_gammas = np.tile(initial_params[n_prev:], (n_pert + 1, 1))

    if "zeros" in initialisation or "first" in initialisation and p > 1:
        remaining_betas = np.zeros(n_remaining)
        remaining_gammas = np.zeros(n_remaining)
    elif "random" in initialisation:
        if fourier:  # initialise in [0, 1]
            remaining_betas = rng.random((1, n_remaining))
            remaining_gammas = rng.random((1, n_remaining))
        else:  # initialise in [-pi/2, pi/2] for beta and [0, pi] for gamma
            remaining_betas = rng.random((1, n_remaining)) * np.pi - 0.5 * np.pi
            remaining_gammas = rng.random((1, n_remaining)) * np.pi
    else:
        raise ValueError(f"Invalid initialisation strategy: {initialisation}")
    remaining_betas = np.tile(remaining_betas, (n_pert + 1, 1))
    remaining_gammas = np.tile(remaining_gammas, (n_pert + 1, 1))

    if initial_params is None:
        remaining_betas, remaining_gammas = compute_random_param_perturbations(
            remaining_betas, remaining_gammas, n_pert, rng
        )
    else:
        prev_betas, prev_gammas = compute_random_param_perturbations(
            prev_betas, prev_gammas, n_pert, rng
        )

    beta_init = np.concatenate([prev_betas, remaining_betas], axis=1)
    gamma_init = np.concatenate([prev_gammas, remaining_gammas], axis=1)

    init_params = np.concatenate([beta_init, gamma_init], axis=1)

    return init_params


def get_parameter_bounds(p: int) -> List[Tuple[float, float]]:
    """
    Computes parameter bounds for QAOA

    Parameters
    ----------
    p : int
        The number of layers of the QAOA circuit.

    Returns
    -------
    List[Tuple[float, float]]
        List of bounds of the parameter space, Tuples with (min, max)
    """

    bounds_beta = (-0.5 * np.pi, 0.5 * np.pi)
    bounds_gamma = (-np.pi, np.pi)
    bounds = [bounds_beta] * p + [bounds_gamma] * p

    return bounds


def get_FOURIER_params(
    v_params: np.ndarray,
    u_params: np.ndarray,
    p: int,
    q: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the parameters from the FOURIER strategy to the QAOA parameters.

    Parameters
    ----------
    v_params : np.ndarray
        The parameters of the cosine part of the FOURIER strategy.
    u_params : np.ndarray
        The parameters of the sine part of the FOURIER strategy.
    p : int
        The number of layers of the QAOA circuit.
    q : int, optional
        The number of parameters in the FOURIER strategy. Defaults to -1.

    Returns
    -------
    betas : np.ndarray
        The QAOA parameters for the beta angles.
    gammas : np.ndarray
        The QAOA parameters for the gamma angles.
    """
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


def spsa(
    fun: callable,
    x0: np.ndarray,
    args: tuple,
    options: Dict,
    tol: Optional[float] = None,
    bounds: Optional[list] = None,
) -> OptimizeResult:
    """
    Perform the Simultaneous Perturbation Stochastic Approximation (SPSA) optimization.
    Implementation with help from
    https://www.geeksforgeeks.org/spsa-simultaneous-perturbation-stochastic-approximation-algorithm-using-python/

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : np.ndarray
        Initial guess for the parameters.
    args : tuple
        Additional arguments to be passed to the objective function.
    tol : Optional[float]
        Tolerance for the optimization process.
    bounds : Optional[list]
        Bounds for the parameters.
    options : Dict
        Options for the optimization process.
        maxiter: number of iterations, defaults to 1000
        alpha: learning rate, defaults to 0.5
        gamma: amplitude of the perturbation, defaults to 0.1
        c: amplitude of the perturbation, defaults to 1e-2
        seed: seed for the random number generator

    Returns
    -------
    OptimizeResult
        The result of the optimization process.
    """
    w = x0
    maxiter = options.get("maxiter", 200)
    alpha = options.get("alpha", 0.722)
    gamma = options.get("gamma", 0.722)
    c = options.get("c", 0.08)
    seed = options.get("seed", 1000)

    rng = np.random.default_rng(seed=seed)

    if bounds is not None:
        assert isinstance(bounds, list) or isinstance(
            bounds, Bounds
        ), "Bounds must be a list or a Bounds object."

    def grad(f_cost, w, c, bounds, args):
        # bernoulli-like distribution
        deltak = rng.choice([-1, 1], size=len(w))

        # simultaneous perturbations
        ck_deltak = c * deltak

        # TODO: this may cause the optimizer to get stuck
        if bounds is not None:
            for i, bound in enumerate(bounds):
                ck_deltak[i] = np.clip(ck_deltak[i], bound[0], bound[1])

        # gradient approximation
        DELTA_L = f_cost(w + ck_deltak, *args) - f_cost(w - ck_deltak, *args)

        return (DELTA_L) / (2 * ck_deltak)

    def initialize_hyperparameters(f_cost, w, alpha, N, bounds, args):

        # A is <= 10% of the number of iterations
        A = N * 0.1

        # order of magnitude of first gradients
        g0_abs = np.abs(grad(f_cost=f_cost, w=w, c=c, bounds=bounds, args=args).mean())

        # the number 2 in the front is an estimative of
        # the initial changes of the parameters,
        # different changes might need other choices
        # Added 1e-3 to avoid division by zero
        a = 2 * ((A + 1) ** alpha) / (g0_abs + 1e-3)

        return a, A

    a, A = initialize_hyperparameters(
        f_cost=fun, w=w, alpha=alpha, N=maxiter, bounds=bounds, args=args
    )

    message = ""
    success = np.True_
    try:
        for k in range(1, maxiter):

            # update ak and ck
            ak = a / ((k + A) ** (alpha))
            ck = c / (k ** (gamma))

            if tol is not None:
                if ak < tol:
                    message = f"Tolerance {tol} reached after {k} iterations."
                    status = 0
                    break

            # estimate gradient
            gk = grad(f_cost=fun, w=w, c=ck, bounds=bounds, args=args)

            # update parameters
            w -= ak * gk
        message = f"Max. iterations ({maxiter}) reached."
        status = 1
    except Exception as e:
        message = f"Terminated with error after {k} iterations: {e}"
        success = np.False_
        status = 2

    return OptimizeResult(
        x=w,
        fun=fun(w, *args) if status < 2 else np.nan,
        nit=k,
        success=success,
        status=status,
        message=message,
    )


def minimize(*args, options, **kwargs):
    if kwargs["method"] == "SPSA":
        del kwargs["method"]
        return spsa(*args, options=options, **kwargs)
    else:
        return scipy_minimize(*args, options=options, **kwargs)


def solve_QUBO_with_QAOA(
    qubo: np.ndarray,
    p: int,
    q: int,
    n_pert: int,
    seed: int,
    initialisation: str,
    parameter_rng: np.random.Generator,
    initial_params: Optional[np.ndarray] = None,
    optimiser: str = "COBYLA",
    tolerance: float = 1e-3,
    maxiter: int = 1000,
    apply_bounds: bool = False,
    options: Dict[str, Any] = {},
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve a QUBO using the QAOA algorithm.

    Parameters
    ----------
    qubo : np.ndarray
        A QUBO matrix.
    p : int
        The number of layers of the QAOA circuit.
    q : int
        The number of parameters in the FOURIER strategy.
    n_pert : int
        The number of random perturbations for the FOURIER strategy.
    seed : int
        The seed for the random number generator.
    initialisation : str
        Initialisation strategy for QAOA parameters.
    parameter_rng : np.random.Generator
        The random number generator for random initialisation.
    initial_params : Optional[np.ndarray], optional
        The initial parameters for the QAOA algorithm. Defaults to None.
    optimiser : str, optional
        The optimiser to use. Defaults to "COBYLA".
    tolerance : float, optional
        The tolerance for the optimization algorithm. Defaults to 1e-3.
    maxiter : int, optional
        Number of maximum iterations for the optimization algorithm. Defaults
        to 1000.
    apply_bounds : bool, optional
        Whether parameter bounds should be applied during optimisation.
        Defaults to False.
    options : Dict, optional
        Additional options for the optimiser. Defaults to empty dict.

    Returns
    -------
    energy : float
        The energy of the solution.
    betas : np.ndarray
        The QAOA parameters for the beta angles.
    gammas : np.ndarray
        The QAOA parameters for the gamma angles.
    u_params : np.ndarray
        The parameters for the U gates.
    v_params : np.ndarray
        The parameters for the V gates.
    """
    H, o = hamiltonian_from_qubo(qubo)
    circ = QAOAAnsatz(cost_operator=H, reps=p)
    estimator = Estimator(seed=seed)
    init_params = initialise_QAOA_parameters(
        p if q == -1 else q,
        initialisation,
        parameter_rng,
        initial_params,
        fourier=q > 0,
        n_pert=n_pert,
    )

    def cost_fkt(
        params: np.ndarray,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: Estimator,
        p: int,
        q: int,
        apply_bounds: bool,
    ) -> float:

        if q != -1:
            v_params, u_params = (
                params[: len(params) // 2],
                params[len(params) // 2 :],
            )
            betas, gammas = get_FOURIER_params(v_params, u_params, p, q)
            betas, gammas = normalise_params(betas, gammas, apply_bounds)
            qaoa_params = np.concatenate([betas, gammas])
        else:
            qaoa_params = params

        qaoa_circ = ansatz.assign_parameters(qaoa_params)
        job = estimator.run([(qaoa_circ, hamiltonian)])
        result = job.result()[0]
        cost = result.data.evs
        return cost

    bounds = get_parameter_bounds(p) if apply_bounds and q == -1 else None

    best_qaoa_energy = np.inf
    v_params, u_params = np.array(()), np.array(())
    betas, gammas = init_params[:, :p], init_params[:, p:]
    for i, x in enumerate(init_params):
        min_result = minimize(
            cost_fkt,
            x,
            args=(circ, H, estimator, p, q, apply_bounds),
            method=optimiser,
            tol=tolerance,
            bounds=bounds,
            options=dict(
                maxiter=maxiter,
                seed=seed,
            )
            | options,
        )
        qaoa_energy = min_result.fun + o
        if qaoa_energy < best_qaoa_energy:
            log.info(
                f"Current best parameter set is at index {i} with energy "
                f"{qaoa_energy}",
            )
            best_qaoa_energy = qaoa_energy
            if q == -1:
                v_params, u_params = np.array(()), np.array(())
                betas, gammas = min_result.x[:p], min_result.x[p:]
            else:
                v_params, u_params = min_result.x[:q], min_result.x[q:]
                betas, gammas = get_FOURIER_params(v_params, u_params, p, q)

    return best_qaoa_energy, betas, gammas, u_params, v_params


def times_from_QAOA_params(betas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    Derive midpoints of each time_interval (gamma_i + beta_i).

    Parameters
    ----------
    betas : np.ndarray
        Beta parameters.
    gammas : np.ndarray
        Gamma parameters.

    Returns
    -------
    np.ndarray
        Array of times.
    """
    p: int = len(betas)  # Number of QAOA layers
    time: float = 0.0  # Initialize total time
    time_midpoints: np.ndarray = np.zeros(p + 1)  # Array to store time midpoints
    for i in range(p):
        time_segment: float = np.abs(gammas[i]) + np.abs(
            betas[i]
        )  # Duration of current segment
        time += time_segment  # Increment total time
        time_midpoints[i] = (
            time - 0.5 * time_segment
        )  # Calculate midpoint of current segment

    time_midpoints[p] = time  # Add total annealing time
    return time_midpoints


def annealing_schedule_from_QAOA_params(
    betas: np.ndarray, gammas: np.ndarray, apply_bounds: bool = False
) -> List[Tuple[float, float]]:
    """
    Derive Annealing schedule from QAOA parameters according to Zhou et al.
    (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.021067, Sec. VB)

    Parameters
    ----------
    betas : np.ndarray
        Beta parameters
    gammas : np.ndarray
        Gamma parameters
    apply_bounds : bool, optional
        Whether parameter bounds should be applied during optimisation.
        Defaults to False.

    Returns
    -------
    List[Tuple[float, float]]
        Annealing schedule (e.g. for Dwave), contains a list of tuples of
        the form (anneal_time, anneal_fraction).
    """
    p = len(betas)
    betas, gammas = normalise_params(betas, gammas, apply_bounds)
    times = times_from_QAOA_params(betas, gammas)
    anneal_schedule = [(0.0, 0.0)]
    for i in range(p):
        anneal_fraction = np.abs(gammas[i]) / (np.abs(gammas[i]) + np.abs(betas[i]))
        anneal_schedule.append((times[i], anneal_fraction))

    # at the end of the anneal schedule, the annealing fraction is 1
    anneal_schedule.append((times[p], 1.0))
    return anneal_schedule


def run_QAOA(
    qubo: np.ndarray,
    seed: int,
    max_p: int,
    q: int,
    n_pert: int,
    optimiser: str,
    tolerance: float,
    maxiter: int,
    apply_bounds: bool,
    initialisation: str,
    options: Dict,
) -> List[dict]:
    """
    Run the QAOA algorithm on a given QUBO problem.

    Parameters
    ----------
    qubo : np.ndarray
        The QUBO matrix to be solved.
    seed : int
        Random seed for reproducibility.
    max_p : int
        Maximum number of QAOA layers to run.
    q : int
        Number of parameters in the FOURIER strategy.
    n_pert : int
        The number of random perturbations for the FOURIER strategy.
    optimiser : str
        The optimiser to use.
    tolerance : float
        The tolerance for the optimization algorithm.
    maxiter : int
        Number of maximum iterations for the optimization algorithm.
    apply_bounds : bool
        Whether parameter bounds should be applied during optimisation.
    initialisation: str
        Initialisation strategy for QAOA parameters.
    options : Dict
        Additional options for the optimiser.

    Returns
    -------
    List[dict]
        A list of results for each layer p, containing QAOA energy,
        beta, gamma, u, and v parameters.
    """
    results: List[dict] = []

    init_params: Optional[np.ndarray] = None
    parameter_rng: np.random.Generator = np.random.default_rng(seed)

    for p in range(1, max_p + 1):
        log.info(f"Running QAOA for q = {q}, p = {p}/{max_p}, R = {n_pert}")

        res: Dict = {"p": p}
        res["qaoa_energy"], betas, gammas, us, vs = solve_QUBO_with_QAOA(
            qubo,
            p,
            q if q <= p else p,
            n_pert,
            seed=seed,
            initial_params=init_params,
            parameter_rng=parameter_rng,
            initialisation=initialisation,
            optimiser=optimiser,
            tolerance=tolerance,
            maxiter=maxiter,
            apply_bounds=apply_bounds,
            options=options,
        )

        log.info(f"QAOA energy: {res['qaoa_energy']}")

        if "all" not in initialisation:
            if q == -1:
                init_params = np.concatenate([betas, gammas])
            else:
                init_params = np.concatenate([vs, us])

        res.update({f"beta{i+1:02d}": b for i, b in enumerate(betas)})
        res.update({f"gamma{i+1:02d}": g for i, g in enumerate(gammas)})
        res.update({f"u{i+1:02d}": u for i, u in enumerate(us)})
        res.update({f"v{i+1:02d}": v for i, v in enumerate(vs)})
        results.append(res)

    return results


def normalise_params(
    betas: np.ndarray,
    gammas: np.ndarray,
    apply_bounds: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalises QAOA Parameters according to bounds and symmetries:
    Resulting parameters are in [0, pi] for gamma and in [-0.5*pi, 0.5*pi] for beta.

    Parameters
    ----------
    betas : np.ndarray
        Beta parameters
    gammas : np.ndarray
        Gamma parameters
    apply_bounds : bool, optional
        Whether to apply bounds. Defaults to False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (normalised betas, normalised gammas)
    """
    neg_gamma_indices = gammas < 0

    # Ensure positive gamma (point symmetry)
    betas[neg_gamma_indices] *= -1
    gammas[neg_gamma_indices] *= -1

    if apply_bounds:
        gammas[gammas > np.pi] = np.pi
        betas[betas < -0.5 * np.pi] = -0.5 * np.pi
        betas[betas > 0.5 * np.pi] = 0.5 * np.pi
    else:
        # Circular Bounds
        betas %= np.pi
        gammas %= 2 * np.pi
    return betas, gammas


def compute_random_param_perturbations(
    prev_beta: np.ndarray,
    prev_gamma: np.ndarray,
    n_pert: int,
    rng: np.random.Generator,
    alpha: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes parameters with random perturbations as described in
    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.021067, Apx. B2)

    Parameters
    ----------
    prev_beta : np.ndarray
        beta-parameters from which to compute the perturbations.
    prev_gamma : np.ndarray
        gamma-parameters from which to compute the perturbations.
    n_pert : int
        Number of perturbations.
    rng : np.random.Generator
        Random number generator.
    alpha : float, optional
        Perturbation strength. Defaults to 0.6.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (
            original beta parameters and perturbations with shape
                [r+1, num_beta_params],
            original gamma parameters and perturbations with shape
                [r+1, num_gamma_params]
        )
    """
    random_perturbations_beta = np.tile(prev_beta[0], (n_pert + 1, 1))
    random_perturbations_gamma = np.tile(prev_gamma[0], (n_pert + 1, 1))

    for i in range(1, n_pert + 1):
        random_perturbations_beta[i] += alpha * rng.normal(0, np.abs(prev_beta[0]))
        random_perturbations_gamma[i] += alpha * rng.normal(0, np.abs(prev_gamma[0]))
    return random_perturbations_beta, random_perturbations_gamma

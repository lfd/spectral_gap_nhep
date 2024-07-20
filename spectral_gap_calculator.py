import numpy as np
import json
import os
import pathlib
import csv

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms.eigensolvers import NumPyEigensolver

from qaoa_utils import provide_random_maxcut_QUBO, hamiltonian_from_qubo


def save_to_csv(data, path, filename):

    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True)

    f = open(path + "/" + filename, "a", newline="")
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def load_data(path, filename):
    datafile = os.path.abspath(path + "/" + filename)
    if os.path.exists(datafile):
        with open(datafile, "rb") as file:
            return json.load(file)


def load_all_results(path):
    if not os.path.isdir(path):
        return []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for datafile in onlyfiles:
        with open(path + "/" + datafile, "rb") as file:
            data.append(json.load(file))
    return data


def save_data(data, path, filename):
    print(path)
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True)

    datafile = os.path.abspath(path + "/" + filename)
    mode = "a" if os.path.exists(datafile) else "w"
    with open(datafile, mode) as file:
        json.dump(data, file)


def build_mixing_hamiltonian(num_qubits):
    # As in https://qiskit.org/documentation/_modules/qiskit/circuit/library/n_local/qaoa_ansatz.html#QAOAAnsatz
    mixer_terms = [
        ("I" * left + "X" + "I" * (num_qubits - left - 1), 1)
        for left in range(num_qubits)
    ]
    mixer_hamiltonian = SparsePauliOp.from_list(mixer_terms)
    return mixer_hamiltonian


# Builds the interpolation hamiltonian for the given problem hamiltonian, fraction s and amount of qubits
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
def calculate_spectral_gap(fraction: float, qubo: np.ndarray, num_dec_pos: int=4):
    num_qubits = len(qubo)
    H = build_hamiltonian(qubo, fraction, num_qubits)
    counter = 0
    eigenvalues = []
    eigenstates = []
    while len(set(eigenvalues)) < 2 and counter != num_qubits:
        # Increase the counter in every iteration, such that the number of searched eigenvalues exponentially increases
        # as long as no two unique eigenvalues are found
        counter = counter + 1
        eigensolver = NumPyEigensolver(k=pow(2, counter))
        eigensolver_result = eigensolver.compute_eigenvalues(H)
        eigenstates = eigensolver_result.eigenstates
        eigenvalues = np.real(eigensolver_result.eigenvalues)
        eigenvalues = np.around(eigenvalues, num_dec_pos)

    eigenvalues = np.real(np.unique(eigenvalues))
    spectral_gap = np.around(np.abs(eigenvalues[0] - eigenvalues[1]), 4)
    return eigenvalues[0], eigenvalues[1], spectral_gap


def main(
    num_qubits, density, seed, fraction, result_path_prefix, include_header=True
):
    csv_data_list = []
    if include_header:
        csv_data_list.append(
            [
                "problem",
                "num_qubits",
                "density",
                "seed",
                "fraction",
                "gs",
                "fes",
                "gap",
            ]
        )

    qubo = provide_random_maxcut_QUBO(num_qubits, density, seed)

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap(fraction, qubo)
        csv_data_list.append(
            [
                "maxcut",
                num_qubits,
                density,
                seed,
                np.round(fraction, 2),
                gs_energy,
                fes_energy,
                gap,
            ]
        )

    for csv_data in csv_data_list:
        save_to_csv(csv_data, result_path_prefix, "spectral_gap_evolution.csv")


if __name__ == "__main__":
    seed = 777
    result_path_prefix = "results/MAXCUT/"
    first = True
    for n in range(4, 19):
        for density in np.linspace(0.5, 1, num=6, endpoint=True):
            fractions = np.linspace(0, 1, num=16, endpoint=True)
            main(n, density, seed, fractions, result_path_prefix, include_header=first)
            first = False

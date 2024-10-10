from os import path as op
import pickle
from dimod.binary import BinaryQuadraticModel
import numpy as np

from hepqpr.qallse.dsmaker import create_dataset
from hepqpr.qallse.cli.func import build_model, solve_neal, print_stats, diff_rows, process_response
from hepqpr.qallse import dumper
from hepqpr.qallse.data_wrapper import DataWrapper
from hepqpr.qallse.plotting import iplot_results, iplot_results_tracks

from qallse_wrapper.model import QallseSplit
from spectral_gap_calculator import calculate_spectral_gap_ising, save_to_csv
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

def to_ising_matrix(ising_quadratic, ising_linear):
    names = []
    for quadratic in ising_quadratic.keys():
        names.append(quadratic[0])
        names.append(quadratic[1])

    names.extend(list(ising_linear.keys()))
    names = list(set(names))

    h = np.zeros(len(names))
    J = np.zeros((len(names), len(names)))

    for k, v in ising_linear.items():
        h[names.index(k)] = v

    for k, v in ising_quadratic.items():
        J[names.index(k[0])][names.index(k[1])] = v

    return J, h

def spectral_gap(
    ising_J, ising_h, fractions, result_path_prefix, geometric_index=0, include_header=True
):
    csv_data_list = []
    if include_header:
        csv_data_list.append(
            [
                "problem",
                "num_qubits",
                "geometric_index",
                "seed",
                "fraction",
                "gs",
                "fes",
                "gap",
            ]
        )

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap_ising(fraction, ising_J, ising_h)
        csv_data_list.append(
            [
                "track reconstruction",
                len(ising_h),
                geometric_index,
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
    output_path = "/tmp"
    prefix = "mini10"
    seed = 12345

    ## Create Data
    metadata, path = create_dataset(
        density=0.1,
        output_path=output_path,
        prefix=prefix,
        gen_doublets=True,
        random_seed=seed,
    )

    ## Build QUBO
    dw = DataWrapper.from_path(path)
    header=True

    for i in range(64):
        extra_config = {"geometric_index" : i}
        model = QallseSplit(dw, **extra_config)
        build_model(path, model, False)
        dumper.dump_model(model, output_path, prefix, qubo_kwargs=dict(w_marker=None, c_marker=None))
        qubo_path = op.join(output_path, prefix + "qubo.pickle")
        print('Wrote qubo to', qubo_path)

        with open(qubo_path, 'rb') as f:
            Q = pickle.load(f)

        # Convert to Ising
        qubo_model = BinaryQuadraticModel.from_qubo(Q)
        ising_linear, ising_quadratic, ising_offset = qubo_model.to_ising()
        ising_J, ising_h = to_ising_matrix(ising_quadratic, ising_linear)
        print(ising_J, ising_h)

        if len(ising_h) > 18:
            print(f"Too many variables for index {i}")
            continue

        result_path_prefix = "results/TR/"
        fractions = np.linspace(0, 1, num=11, endpoint=True)
        spectral_gap(ising_J, ising_h, fractions, result_path_prefix, geometric_index=i, include_header=header)
        header=False

        ## Solve QUBO with simulated annealer
        response = solve_neal(Q, seed=seed)
        print_stats(dw, response, Q)
        oname = op.join(output_path, prefix + "neal_response.pickle")
        with open(oname, 'wb') as f: pickle.dump(response, f)
        print(f'Wrote response to {oname}')

        ## Plot

        dims = list("xy")

        final_doublets, final_tracks = process_response(response)
        _, missings, _ = diff_rows(final_doublets, dw.get_real_doublets())
        dout = op.join(output_path, prefix + "plot-doublets.html")
        iplot_results(dw, final_doublets, missings, dims=dims, filename=dout)

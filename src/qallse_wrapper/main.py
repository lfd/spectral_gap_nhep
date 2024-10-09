from os import path as op
import pickle

from hepqpr.qallse.dsmaker import create_dataset
from hepqpr.qallse.cli.func import build_model, solve_neal, print_stats, diff_rows, process_response
from hepqpr.qallse import dumper
from hepqpr.qallse.data_wrapper import DataWrapper
from hepqpr.qallse.plotting import iplot_results, iplot_results_tracks

from qallse_wrapper.model import QallseSplit

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
    extra_config = {}
    model = QallseSplit(dw, **extra_config)
    build_model(path, model, False)
    dumper.dump_model(model, output_path, prefix, qubo_kwargs=dict(w_marker=None, c_marker=None))
    qubo_path = op.join(output_path, prefix + "qubo.pickle")
    print('Wrote qubo to', qubo_path)

    ## Solve QUBO with simulated annealer
    with open(qubo_path, 'rb') as f:
        Q = pickle.load(f)

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
    tout = op.join(output_path, prefix + "plot-triplets.html")
    iplot_results(dw, final_doublets, missings, dims=dims, filename=dout)

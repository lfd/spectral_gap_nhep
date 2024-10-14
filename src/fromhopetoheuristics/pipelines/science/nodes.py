import pickle
from qallse.cli.func import (
    build_model,
    solve_neal,
    print_stats,
)
from qallse import dumper
from qallse.data_wrapper import DataWrapper
from fromhopetoheuristics.utils.model import QallseSplit
import os
import logging

log = logging.getLogger(__name__)


def build_qubo(event_path, output_path, prefix):
    dw = DataWrapper.from_path(event_path)
    extra_config = {}
    model = QallseSplit(dw, **extra_config)
    build_model(event_path, model, False)
    dumper.dump_model(
        model, output_path, prefix, qubo_kwargs=dict(w_marker=None, c_marker=None)
    )
    qubo_path = os.path.join(output_path, prefix + "qubo.pickle")
    print("Wrote qubo to", qubo_path)

    return {"qubo_path": qubo_path}


def solve_qubo(event_path, qubo_path, output_path, prefix, seed):
    dw = DataWrapper.from_path(event_path)

    with open(qubo_path, "rb") as f:
        Q = pickle.load(f)

    response = solve_neal(Q, seed=seed)
    print_stats(dw, response, Q)
    oname = os.path.join(output_path, prefix + "neal_response.pickle")
    with open(oname, "wb") as f:
        pickle.dump(response, f)
    print(f"Wrote response to {oname}")

    return {"response": response}

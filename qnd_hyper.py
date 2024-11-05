import csv
import glob
import os
import numpy as np
import itertools

hyperparams = dict(
    alpha=np.linspace(0.1, 0.8, 10),
    gamma=np.linspace(0.1, 0.8, 10),
    c=np.linspace(0.01, 0.08, 10),
)


def get_data():
    # get the experiment folders
    experiments = glob.glob("data/05_qaoa/qaoa_maxcut_results.csv/**")

    # find the latest experiment based on name as timestamp
    latest = max(experiments, key=os.path.getctime)

    # read the csv file
    with open(f"{latest}/qaoa_maxcut_results.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    energies = []
    for row in data[1:]:
        energies.append(float(row[1]))
    return np.mean(energies)


def run_experiment(alpha, gamma, c):
    # run kedro experiment from commandline and pass parameters
    print("Running...")
    os.system(
        f"kedro run --pipeline gs --params=options.alpha={alpha},options.gamma={gamma},options.c={c}"
    )


def main():
    results = []
    for alpha in hyperparams["alpha"]:
        for gamma in hyperparams["gamma"]:
            for c in hyperparams["c"]:
                run_experiment(alpha, gamma, c)
                mean_energy = get_data()
                print(
                    f"alpha: {alpha:.4}, gamma: {gamma:.4}, c: {c:.4} -> {mean_energy}"
                )
                results.append(
                    {"alpha": alpha, "gamma": gamma, "c": c, "mean_energy": mean_energy}
                )

    print(results)
    with open("hp_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "gamma", "c", "mean_energy"])
        for result in results:
            writer.writerow(
                [result["alpha"], result["gamma"], result["c"], result["mean_energy"]]
            )


if __name__ == "__main__":
    main()

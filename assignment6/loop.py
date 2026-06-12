import sys
import os
import re
import subprocess
import numpy as np
import time
from statistics import mean, stdev

EPOCHS = 300
SEEDS = [0, 1, 2, 3, 4]
CARBON_INTENSITY = 200.0

LAYERS = [1, 2, 3]
NEURONS = [8, 16, 24, 32, 40, 48, 56, 64]

#Subprocess logic implemented by Claude Opus 4.8
USER = os.environ.get("USER", "ubuntu")
TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainer.py")

#Subprocess logic implemented by Claude Opus 4.8
def measure_energy(layers, neurons, seed):
    start = time.time()
    command = [
        "sudo", "turbostat", "-q", "-S", "--Joules", "--show", "Pkg_J",
        "sudo", "-u", USER, sys.executable, TRAINER,
        str(layers), str(neurons), str(seed),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    end = time.time()
    output = result.stdout + result.stderr

    accuracy = float(re.search(r"ACC ([\d.]+)", output).group(1))

    joules = None
    lines = output.splitlines()
    for i in range(len(lines) - 1):
        header = lines[i].split()
        values = lines[i + 1].split()
        if "Pkg_J" in header:
            column = header.index("Pkg_J")
            joules = float(values[column])
            break

    return joules, accuracy, end - start


results = []
for layers in LAYERS:
    for neurons in NEURONS:
        energies = []
        accuracies = []
        times = []
        for seed in SEEDS:
            joules, accuracy, elapsed_time = measure_energy(layers, neurons, seed)
            energies.append(joules)
            accuracies.append(accuracy)
            times.append(elapsed_time)

        average_energy = mean(energies)
        average_time = mean(times)
        results.append({
            "layers": layers,
            "neurons": neurons,
            "accuracy": mean(accuracies),
            "accuracy_std": stdev(accuracies),
            "energy": average_energy,
            "energy_std": stdev(energies),
            "energy_per_epoch": average_energy / EPOCHS,
            "carbon": average_energy / 3.6e6 * CARBON_INTENSITY,
            "time": average_time,
        })


print("layers neurons   accuracy            energy(J)        energy per epoch     gCO2     time(s)")
for r in results:
    print(
        "  ", r["layers"],
        "    ", r["neurons"],
        "   ", round(r["accuracy"], 3), "+/-", round(r["accuracy_std"], 3),
        "  ", round(r["energy"], 2), "+/-", round(r["energy_std"], 2),
        "  ", round(r["energy_per_epoch"], 4),
        "  ", round(r["carbon"], 4),
        "  ", round(r["time"], 4),
    )

#"Pareto front" logic implemented as pseudocode by Claude Opus 4.8
pareto_front = []
for r in results:
    is_dominated = False
    for other in results:
        more_accurate = other["accuracy"] >= r["accuracy"]
        less_energy = other["energy"] <= r["energy"]
        strictly_better = (other["accuracy"] > r["accuracy"]) or (other["energy"] < r["energy"])
        if more_accurate and less_energy and strictly_better:
            is_dominated = True
            break
    if not is_dominated:
        pareto_front.append(r)

print("\nPareto front (best trade-offs):")
for r in pareto_front:
    print("  ", r["layers"], "layers,", r["neurons"], "neurons")

best = None
for r in pareto_front:
    if r["accuracy"] >= 0.96 and r["energy"] <= 40:
        if best is None or r["energy"] < best["energy"]:
            best = r

if best is None:
    print("\nNo config met the requirements.")
else:
    print("\nBest choice:", best["layers"], "layers,", best["neurons"], "neurons")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for layers in LAYERS:
    x_values = []
    y_values = []
    for r in results:
        if r["layers"] == layers:
            x_values.append(r["neurons"])
            y_values.append(r["energy_per_epoch"])

    points = plt.scatter(x_values, y_values, label=str(layers) + " layer(s)")
    color = points.get_facecolor()[0]

    a, b, c = np.polyfit(x_values, y_values, 2)

    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = a * x_line * x_line + b * x_line + c

    plt.plot(x_line, y_line, color=color)
    print(layers, "layer(s): a =", a, " b =", b, " c =", c)

plt.xlabel("neurons per layer")
plt.ylabel("training energy per epoch (J)")
plt.title("Energy per epoch vs number of neurons")
plt.legend()
plt.savefig("energy_vs_neurons.png")
print("\nsaved energy_vs_neurons.png")


"""E49's figure: three trainers that were not written for PlugRL, training PlugRL env clients.

    python figure.py OUT.png      # anywhere with matplotlib

Reads each registered run's episode log, results/registered/<cell>-seed<s>/episodes.csv, which
the bridge writes: one row per finished episode (step, slot, episode, return, length), where step
counts vector steps of the 16 env clients. Plotted: every seed's mean return over the 16 clients'
k-th episodes, against environment steps (vector steps x 16). No mean line and no smoothing.
"""

import csv
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "results" / "registered"
CLIENTS = 16
# (panel title, [(cell, trainer label, colour)]); Okabe-Ito colours, one per trainer.
PANELS = [
    (
        "Pendulum, C++ env client",
        [
            ("rlinf-pendulum", "RLinf", "#D55E00"),
            ("sb3-pendulum", "Stable-Baselines3", "#0072B2"),
        ],
    ),
    (
        "HalfCheetah-v5, Python env client",
        [
            ("sb3-halfcheetah", "Stable-Baselines3", "#0072B2"),
            ("cleanrl-halfcheetah", "CleanRL", "#009E73"),
        ],
    ),
]


def curve(path: pathlib.Path) -> tuple[list[float], list[float]]:
    """(environment steps in thousands, mean return) per round of the clients' episodes."""
    rounds = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rounds[int(r["episode"])].append((int(r["step"]), float(r["return"])))
    xs, ys = [], []
    for k in sorted(rounds):
        if len(rounds[k]) < CLIENTS:  # a round some client had not finished
            continue
        xs.append(max(s for s, _ in rounds[k]) * CLIENTS / 1000)
        ys.append(sum(v for _, v in rounds[k]) / CLIENTS)
    return xs, ys


def plot(out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams["svg.hashsalt"] = "e49"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200)
    trainers = {}
    for ax, (title, cells) in zip(axes, PANELS):
        for cell, label, colour in cells:
            for seed in range(3):
                xs, ys = curve(RUNS / f"{cell}-seed{seed}" / "episodes.csv")
                ax.plot(xs, ys, color=colour, linewidth=0.9, alpha=0.8)
                print(
                    f"{cell} seed {seed}: {ys[0]:.0f} -> {ys[-1]:.0f} over {xs[-1]:.1f}k steps"
                )
            trainers[label] = colour
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("environment steps (thousands)")
        ax.set_xlim(0, None)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e3e6f0", linewidth=0.6)
    axes[0].set_ylabel("episode return")
    # One legend for both panels: a trainer keeps its colour in each.
    handles = [
        Line2D([], [], color=c, linewidth=2, label=f"{t} PPO")
        for t, c in trainers.items()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=3,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor="white", metadata={"Software": None})
    print("wrote", out)


if __name__ == "__main__":
    plot(pathlib.Path(sys.argv[1]))

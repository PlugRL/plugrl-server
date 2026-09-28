"""E43's figure: the same cell trained on one machine and across two.

    python figure.py export [results-dir]   # on guangzhao: tensorboards -> results/curves.json
    python figure.py plot OUT.png           # anywhere with matplotlib

Every seed's return per iteration, over its arm's band from the lowest seed to
the highest. No mean line: with six seeds the arms' means differ by less than
either arm's spread (AMENDMENT.md). The legend carries each arm's range of
wall clock from iteration 1 to 100.
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
CURVES = HERE / "results" / "curves.json"
# Seeds 0-2 are the registered run; 3-5 are AMENDMENT.md's, in the -b directories.
LAYOUT = [(0, ""), (1, ""), (2, ""), (3, "-b"), (4, "-b"), (5, "-b")]


def export(results: pathlib.Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    data = {}
    for arm in ("local", "cross"):
        data[arm] = {"returns": [], "span_s": []}
        for seed, suffix in LAYOUT:
            run = results / f"{arm}{suffix}" / "fpo" / "fpo-policy" / f"halfcheetah-{arm}-seed{seed}"
            if not run.exists():
                continue
            acc = EventAccumulator(str(sorted((run / "tensorboard").glob("events.*"))[-1]),
                                   size_guidance={"scalars": 0})
            acc.Reload()
            r = acc.Scalars("rollout/reward")
            data[arm]["returns"].append([round(e.value, 1) for e in r])
            data[arm]["span_s"].append(round(r[-1].wall_time - r[0].wall_time, 1))
    CURVES.write_text(json.dumps(data) + "\n", encoding="utf-8")
    print("wrote", CURVES)


def plot(out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(CURVES.read_text(encoding="utf-8"))
    style = {
        "local": ("#9aa5d8", "one machine"),
        "cross": ("#4051b5", "two machines, over campus Wi-Fi"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for arm in ("local", "cross"):
        colour, label = style[arm]
        runs = data[arm]["returns"]
        n = min(len(r) for r in runs)
        x = list(range(1, n + 1))
        # The spread across seeds, and each seed: no mean line, since the
        # arms' means differ by less than either arm's spread (AMENDMENT.md).
        lo = [min(r[i] for r in runs) for i in range(n)]
        hi = [max(r[i] for r in runs) for i in range(n)]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
        for r in runs:
            ax.plot(x, r[:n], color=colour, linewidth=0.8, alpha=0.7)
        spans = [s / 60 for s in data[arm]["span_s"]]
        ax.plot([], [], color=colour, linewidth=6, alpha=0.5,
                label=f"{label} ({min(spans):.0f}-{max(spans):.0f} min)")
    ax.set_xlabel("iteration (4,096 steps each)")
    ax.set_ylabel("episode return")
    seeds = len(data["local"]["returns"])
    ax.set_title(f"fpo-policy · FPO · HalfCheetah-v5, {seeds} seeds per arm", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e3e6f0", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE / "results")
    else:
        plot(pathlib.Path(sys.argv[2]))

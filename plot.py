"""Render runs/<run>/log.csv to runs/<run>/reward_curve.png.   python plot.py runs/main"""
import csv
import os
import sys


def plot_run(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(run_dir, "log.csv")) as f:
        rows = list(csv.DictReader(f))

    def series(key):
        pts = [(float(r["agent_steps"]) / 1e6, float(r[key])) for r in rows if r.get(key) not in ("", None)]
        return [p[0] for p in pts], [p[1] for p in pts]

    def smooth(xy, k=15):
        x, y = xy
        if len(y) < k:
            return x, y
        c = [sum(y[max(0, i - k + 1):i + 1]) / len(y[max(0, i - k + 1):i + 1]) for i in range(len(y))]
        return x, c

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(*smooth(series("pool_win")), label="win vs past selves (smoothed)", color="0.7")
    for key, label in (("rule_win", "win vs rule bot"), ("random_win", "win vs random"),
                       ("strafer_win", "win vs strafer bot"), ("strafer_loss", "loss vs strafer bot"),
                       ("v1_win", "win vs v1 Pip (head-to-head)"), ("v1_loss", "loss vs v1 Pip")):
        ax[0, 0].plot(*series(key), label=label)
    ax[0, 0].set_title("Win rates (deterministic eval / self-play pool)")
    ax[0, 0].set_ylim(-0.02, 1.02)
    ax[0, 0].legend()
    ax[0, 1].plot(*series("rule_dodge"), label="dodge rate vs rule bot")
    ax[0, 1].plot(*series("strafer_dodge"), label="dodge rate vs strafer bot (leads shots)")
    ax[0, 1].plot(*series("lateral"), label="lateral movement fraction")
    ax[0, 1].set_title("Behaviour")
    ax[0, 1].legend()
    ax[1, 0].plot(*smooth(series("ep_len_s")), label="self-play episode length (s)")
    ax[1, 0].plot(*series("hits_per_ep"), label="hits per episode (both)")
    ax[1, 0].set_title("Self-play episodes")
    ax[1, 0].legend()
    ax[1, 1].plot(*series("entropy"), label="policy entropy")
    ax[1, 1].plot(*series("draw_rate"), label="self-play draw rate")
    ax[1, 1].set_title("Training")
    ax[1, 1].legend()
    for a in ax.flat:
        a.set_xlabel("agent-steps (M)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "reward_curve.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    plot_run(sys.argv[1] if len(sys.argv) > 1 else "runs/main")

"""
Compare episode reward curves across algorithms (e.g. CT-SAC vs a standard RL
baseline like SB3 SAC).

Both `run_ct_rl` and `run_discrete_rl` write a `progress.csv` into each run
directory containing:
    - time/total_timesteps   (x-axis)
    - rollout/ep_rew_mean    (mean episode return over the last 100 episodes)

This script collects those curves, optionally averages across seeds (mean +/-
std band), and plots them on a single figure.

Examples
--------
Compare one CT-SAC run against one baseline run (point at any directory; it is
searched recursively for progress.csv):

    python -m benchmarks.plot_compare \
        --run "CT-SAC=log_test/ct_sac/pendulum" \
        --run "SAC=log_discrete/sac/pendulum" \
        --out compare_pendulum.png

Multiple seeds under a label are aggregated automatically (mean +/- std):

    python -m benchmarks.plot_compare \
        --run "CT-SAC=log_test/ct_sac/pendulum/top" \
        --run "SAC=log_discrete/sac/pendulum/top"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Column names written by both the CT and discrete runners.
X_KEY = "time/total_timesteps"
Y_KEY = "rollout/ep_rew_mean"


def find_progress_files(path: Path) -> List[Path]:
    """Return all progress.csv files under `path` (or the file itself)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("progress.csv"))
    raise FileNotFoundError(f"No such file or directory: {path}")


def load_curve(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load (timesteps, ep_rew_mean) from one progress.csv, dropping empty rows."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return np.array([]), np.array([])  # empty/partial run, nothing logged yet
    if X_KEY not in df.columns or Y_KEY not in df.columns:
        raise KeyError(
            f"{csv_path} is missing '{X_KEY}' and/or '{Y_KEY}'. "
            f"Available columns: {list(df.columns)}"
        )
    sub = df[[X_KEY, Y_KEY]].dropna()
    sub = sub.sort_values(X_KEY)
    return sub[X_KEY].to_numpy(dtype=float), sub[Y_KEY].to_numpy(dtype=float)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean; returns y unchanged if window <= 1."""
    if window <= 1 or y.size == 0:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().to_numpy()


def aggregate_seeds(
    curves: List[Tuple[np.ndarray, np.ndarray]], num_points: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample each (x, y) curve onto a shared x-grid and return
    (grid, mean, std) across seeds. With a single seed, std is all zeros.
    """
    x_max = min(x[-1] for x, _ in curves)  # don't extrapolate past shortest run
    x_min = max(x[0] for x, _ in curves)
    grid = np.linspace(x_min, x_max, num_points)
    stacked = np.vstack([np.interp(grid, x, y) for x, y in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def parse_run_arg(arg: str) -> Tuple[str, Path]:
    """Parse a 'LABEL=PATH' CLI argument."""
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--run expects 'LABEL=PATH', got {arg!r}"
        )
    label, path = arg.split("=", 1)
    return label.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A run to plot. PATH may be a progress.csv or a directory searched "
        "recursively for progress.csv (multiple seeds are averaged).",
    )
    parser.add_argument(
        "--smooth", type=int, default=5, help="Rolling-mean window (1 = off)."
    )
    parser.add_argument(
        "--title", default="Episode reward: CT-SAC vs baseline"
    )
    parser.add_argument("--xlabel", default="Environment steps")
    parser.add_argument("--ylabel", default="Mean episode reward")
    parser.add_argument(
        "--out", default=None, help="Output image path. If omitted, shows the plot."
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))

    for run_arg in args.run:
        label, path = parse_run_arg(run_arg)
        files = find_progress_files(path)
        if not files:
            print(f"[warn] no progress.csv found for '{label}' under {path}")
            continue

        curves = []
        for f in files:
            try:
                x, y = load_curve(f)
            except KeyError as e:
                print(f"[warn] skipping {f}: {e}")
                continue
            if x.size:
                curves.append((x, smooth(y, args.smooth)))

        if not curves:
            print(f"[warn] no usable curves for '{label}'")
            continue

        grid, mean, std = aggregate_seeds(curves)
        line, = ax.plot(grid, mean, label=f"{label} (n={len(curves)})")
        if len(curves) > 1:
            ax.fill_between(
                grid, mean - std, mean + std, color=line.get_color(), alpha=0.2
            )
        print(f"[ok] {label}: {len(curves)} run(s), final reward ~ {mean[-1]:.2f}")

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

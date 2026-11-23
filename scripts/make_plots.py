"""Builds the result tables and figures from runs/ and results/.

Example:
    python scripts/make_plots.py --runs runs --results results --out figures
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["collision", "offroad", "progress", "success", "ep_return",
           "mean_abs_jerk", "mean_abs_dkappa"]
NICE = {"expert": "Logged human", "idm_ego": "IDM (Waymax)",
        "pure_pursuit": "Hand-designed", "balanced": "PPO (balanced)",
        "progress": "PPO (progress)", "safety": "PPO (safety)"}
FAILURES = ["struck_from_behind", "rear_end", "collision_turning", "collision_other",
            "offroad_curve", "offroad_straight", "stalled"]


def load_eval(folder):
    """Returns {method: [DataFrame per seed]} for one traffic setting."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        method = name.rsplit("_s", 1)[0] if "_s" in name else name
        out.setdefault(method, []).append(pd.read_csv(path))
    return out


def summary_table(evals):
    rows = []
    for method, dfs in evals.items():
        per_seed = pd.DataFrame([{m: df[m].mean() for m in METRICS} for df in dfs])
        row = {"method": NICE.get(method, method), "seeds": len(dfs),
               "episodes": len(dfs[0])}
        for m in METRICS:
            row[m] = per_seed[m].mean()
            row[m + "_std"] = per_seed[m].std(ddof=0) if len(dfs) > 1 else np.nan
        rows.append(row)
    order = ["Logged human", "IDM (Waymax)", "Hand-designed", "PPO (balanced)",
             "PPO (progress)", "PPO (safety)"]
    df = pd.DataFrame(rows)
    df["_o"] = df["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def to_markdown(df, title):
    def cell(r, m, pct=True, digits=1):
        v, s = r[m], r[m + "_std"]
        if pct:
            v, s = 100 * v, 100 * s
        txt = f"{v:.{digits}f}"
        return txt + (f" ± {s:.{digits}f}" if not np.isnan(s) else "")

    lines = [f"**{title}**", "",
             "| Method | Seeds | Collision % | Off-road % | Route progress % | Success % | Return | Jerk (m/s³) | Δκ (1/m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['method']} | {r['seeds']} | {cell(r, 'collision')} | {cell(r, 'offroad')} | "
            f"{cell(r, 'progress')} | {cell(r, 'success')} | {cell(r, 'ep_return', False)} | "
            f"{cell(r, 'mean_abs_jerk', False, 2)} | {cell(r, 'mean_abs_dkappa', False, 4)} |")
    return "\n".join(lines)


def learning_curves(runs_dir, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for preset, color in [("balanced", "C0"), ("progress", "C1"), ("safety", "C2")]:
        logs = [pd.read_csv(p) for p in sorted(glob.glob(f"{runs_dir}/{preset}_s*/progress.csv"))]
        if not logs:
            continue
        n = min(len(lg) for lg in logs)
        x = logs[0]["env_steps"].values[:n] / 1e6
        for ax, col, label in zip(axes, ["ep_return", "collision", "progress"],
                                  ["Episode return", "Collision rate", "Route progress"]):
            ys = np.stack([lg[col].rolling(10, min_periods=1).mean().values[:n] for lg in logs])
            m, s = ys.mean(0), ys.std(0)
            ax.plot(x, m, color=color, label=f"{NICE[preset]} (n={len(logs)})")
            if len(logs) > 1:
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
            ax.set_xlabel("Environment steps (M)")
            ax.set_title(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Training curves (stochastic policy, replayed traffic)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning_curves.png"), dpi=150)
    plt.close(fig)


def main_comparison(table, out):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    metrics = ["collision", "offroad", "success"]
    w = 0.8 / len(table)
    for i, (_, r) in enumerate(table.iterrows()):
        vals = [100 * r[m] for m in metrics]
        errs = [100 * r[m + "_std"] if not np.isnan(r[m + "_std"]) else 0 for m in metrics]
        ax.bar(np.arange(3) + i * w, vals, w, yerr=errs, capsize=3, label=r["method"])
    ax.set_xticks(np.arange(3) + w * (len(table) - 1) / 2)
    ax.set_xticklabels(["Collision %", "Off-road %", "Success %"])
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Held-out scenarios, replayed traffic")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "main_comparison.png"), dpi=150)
    plt.close(fig)


def reward_study(table, out):
    fig, ax = plt.subplots(figsize=(5, 4))
    for _, r in table.iterrows():
        x, y = 100 * r["progress"], 100 * r["collision"]
        xe = 100 * r["progress_std"] if not np.isnan(r["progress_std"]) else 0
        ye = 100 * r["collision_std"] if not np.isnan(r["collision_std"]) else 0
        marker = "o" if r["method"].startswith("PPO") else "s"
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=marker, capsize=3, label=r["method"])
    ax.set_xlabel("Route progress (% of logged route)")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Reward study: safety vs progress")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reward_study.png"), dpi=150)
    plt.close(fig)


def failure_breakdown(evals, out):
    rows = {}
    for method, dfs in evals.items():
        df = pd.concat(dfs)
        counts = df["failure"].value_counts(normalize=True)
        rows[NICE.get(method, method)] = [100 * counts.get(f, 0.0) for f in FAILURES]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(rows))
    names = list(rows)
    for j, f in enumerate(FAILURES):
        vals = np.array([rows[n][j] for n in names])
        ax.bar(names, vals, bottom=bottom, label=f.replace("_", " "))
        bottom += vals
    ax.set_ylabel("% of episodes")
    ax.set_title("Failure types (replayed traffic)")
    ax.legend(fontsize=7, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "failure_types.png"), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows, index=FAILURES).round(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    md = []
    replay = load_eval(os.path.join(args.results, "replay"))
    t_replay = summary_table(replay)
    t_replay.to_csv(os.path.join(args.results, "table_replay.csv"), index=False)
    md.append(to_markdown(t_replay, "Replayed traffic"))
    main_comparison(t_replay[~t_replay.method.isin(["PPO (progress)", "PPO (safety)"])], args.out)
    reward_study(t_replay[t_replay.method != "IDM (Waymax)"], args.out)
    fails = failure_breakdown(replay, args.out)
    fails.to_csv(os.path.join(args.results, "failure_types.csv"))
    md += ["", "**Failure types (% of episodes, replayed traffic)**", "",
           fails.to_markdown()]

    idm_dir = os.path.join(args.results, "idm")
    if os.path.isdir(idm_dir):
        t_idm = summary_table(load_eval(idm_dir))
        t_idm.to_csv(os.path.join(args.results, "table_idm.csv"), index=False)
        md += ["", to_markdown(t_idm, "Reactive IDM traffic (same policies, no retraining)")]
    learning_curves(args.runs, args.out)
    with open(os.path.join(args.results, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

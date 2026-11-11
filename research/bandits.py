# Poisson bandits, part of the research note (section 3).
#
# Two experiments:
#   python bandits.py long    -> KL-UCB and Thompson on the project instance (2, 3, 4) up to n = 1e6,
#                                to see if the slope of R_n vs log n really goes to the Lai-Robbins constant
#   python bandits.py scale   -> two arms (mu, mu - 1) with mu = 2, 4, 16, 64. The gap stays 1 but the
#                                variance grows, so UCB with a fixed sigma should break at some point
#
# All runs of one algorithm are simulated together in numpy arrays (shape runs x arms).
# "long" takes ~10 min (KL-UCB is the slow one because of the bisection), "scale" ~4 min.

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

COLORS = {"UCB (sigma=1)": "tab:green", "UCB (oracle sigma)": "tab:red",
          "KL-UCB": "tab:purple", "Thompson": "tab:brown"}


def kl_poisson(p, q):
    p = np.maximum(p, 1e-12)
    return p * np.log(p / q) + q - p


def lai_robbins(mus):
    best = max(mus)
    return sum((best - m) / kl_poisson(m, best) for m in mus if m < best)


def kl_ucb_index(mu_hat, nb, t, n_iter=22):
    # biggest q >= mu_hat with nb * KL(mu_hat, q) <= log t, found by bisection
    level = np.log(t) / nb
    lo = mu_hat.copy()
    hi = mu_hat + 1.0
    for _ in range(60):
        low_still = kl_poisson(mu_hat, hi) < level
        if not low_still.any():
            break
        hi = np.where(low_still, mu_hat + 2 * (hi - mu_hat), hi)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        ok = kl_poisson(mu_hat, mid) <= level
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def run(algo, mus, n, runs, rng, n_grid=250):
    mus = np.asarray(mus, dtype=float)
    K = len(mus)
    gaps = mus.max() - mus
    sigma_oracle = np.sqrt(mus.max())  # variance of the best arm, the largest one
    total = np.zeros((runs, K))
    pulls = np.zeros((runs, K))
    regret = np.zeros(runs)
    rows = np.arange(runs)
    grid = np.unique(np.logspace(0, np.log10(n), n_grid).astype(int))
    curve = np.zeros((len(grid), runs))
    j = 0
    for t in range(1, n + 1):
        if t <= K:
            a = np.full(runs, t - 1)
        else:
            mu_hat = total / pulls
            if algo == "UCB (sigma=1)":
                idx = mu_hat + np.sqrt(2 * np.log(t) / pulls)
            elif algo == "UCB (oracle sigma)":
                idx = mu_hat + np.sqrt(2 * sigma_oracle ** 2 * np.log(t) / pulls)
            elif algo == "KL-UCB":
                idx = kl_ucb_index(mu_hat, pulls, t)
            elif algo == "Thompson":
                idx = rng.gamma(0.5 + total, 1.0 / pulls)  # Jeffreys prior -> Gamma posterior
            else:
                raise ValueError(algo)
            a = np.argmax(idx + 1e-9 * rng.random(idx.shape), axis=1)
        x = rng.poisson(mus[a])
        total[rows, a] += x
        pulls[rows, a] += 1
        regret += gaps[a]
        if j < len(grid) and t == grid[j]:
            curve[j] = regret
            j += 1
    return grid, curve  # curve[time, run]


def slope_fit(grid, mean_curve, lo, hi):
    # least squares R_n = a log n + b on lo <= n <= hi
    m = (grid >= lo) & (grid <= hi)
    a, b = np.polyfit(np.log(grid[m]), mean_curve[m], 1)
    return float(a), float(b)


def long_horizon(n=1_000_000, runs=100):
    mus = [2.0, 3.0, 4.0]
    C = lai_robbins(mus)
    out = {"mus": mus, "n": n, "runs": runs, "C": C}
    plt.figure(figsize=(6.4, 4))
    for i, algo in enumerate(["KL-UCB", "Thompson"]):
        t0 = time.time()
        grid, curve = run(algo, mus, n, runs, np.random.default_rng(100 + i))
        mean = curve.mean(axis=1)
        se = curve.std(axis=1, ddof=1) / np.sqrt(runs)
        a_late, b_late = slope_fit(grid, mean, 1e4, n)
        a_mid, _ = slope_fit(grid, mean, 1e3, 1e4)
        # local slope on each decade, so we see if it is still moving
        dec = {}
        for lo in [1e2, 1e3, 1e4, 1e5]:
            dec[f"{int(lo)}-{int(lo * 10)}"] = slope_fit(grid, mean, lo, lo * 10)[0]
        out[algo] = {"regret_final_mean": float(mean[-1]), "regret_final_se": float(se[-1]),
                     "ratio_final": float(mean[-1] / np.log(n)), "slope_1e4_1e6": a_late,
                     "intercept_1e4_1e6": b_late, "slope_1e3_1e4": a_mid, "slope_by_decade": dec}
        print(algo, out[algo], f"{time.time() - t0:.0f}s", flush=True)
        plt.plot(grid, mean, color=COLORS[algo], label=algo)
        plt.fill_between(grid, mean - 2 * se, mean + 2 * se, color=COLORS[algo], alpha=0.2)
    g = grid[grid >= 3]
    plt.plot(g, C * np.log(g), "k--", lw=1, label=f"{C:.2f} log n (Lai-Robbins)")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("mean regret")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "bandits_long.png"), dpi=200)
    plt.close()
    with open(os.path.join(RES, "bandits_long.json"), "w") as f:
        json.dump(out, f, indent=2)


def scale_family(n=100_000, runs=100):
    tops = [2.0, 4.0, 16.0, 64.0]
    algos = ["UCB (sigma=1)", "UCB (oracle sigma)", "KL-UCB", "Thompson"]
    out = {"n": n, "runs": runs, "instances": {}}
    ratios = {a: [] for a in algos}
    stuck = {a: [] for a in algos}
    for k, top in enumerate(tops):
        mus = [top - 1.0, top]
        C = lai_robbins(mus)
        out["instances"][str(top)] = {"mus": mus, "C": C}
        for i, algo in enumerate(algos):
            t0 = time.time()
            grid, curve = run(algo, mus, n, runs, np.random.default_rng(1000 + 10 * k + i))
            final = curve[-1]
            r = {"mean": float(final.mean()), "median": float(np.median(final)),
                 "se": float(final.std(ddof=1) / np.sqrt(runs)), "max": float(final.max()),
                 "ratio_mean_to_ClogN": float(final.mean() / (C * np.log(n))),
                 "ratio_median_to_ClogN": float(np.median(final) / (C * np.log(n))),
                 "stuck_share": float(np.mean(final > 0.1 * n)),
                 "slope_1e3_1e5": slope_fit(grid, curve.mean(axis=1), 1e3, n)[0]}
            out["instances"][str(top)][algo] = r
            ratios[algo].append(r["ratio_mean_to_ClogN"])
            stuck[algo].append(r["stuck_share"])
            print(top, algo, r, f"{time.time() - t0:.0f}s", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for algo in algos:
        ax[0].plot(tops, ratios[algo], "o-", color=COLORS[algo], label=algo)
        ax[1].plot(tops, np.array(stuck[algo]) * 100, "o-", color=COLORS[algo], label=algo)
    ax[0].axhline(1, color="k", ls="--", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("mean of the best arm (gap is always 1)")
    ax[0].set_ylabel("mean regret / (C log n) at n = 1e5")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("mean of the best arm (gap is always 1)")
    ax[1].set_ylabel("% of runs with regret > n/10")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bandits_scale.png"), dpi=200)
    plt.close(fig)
    with open(os.path.join(RES, "bandits_scale.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("long", "both"):
        long_horizon()
    if which in ("scale", "both"):
        scale_family()

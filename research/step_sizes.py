# Step sizes for TD(0), section 4 of the note.
#
# Everything is on the green robot from the project (exercise 3.1), policy evaluation of s/re,
# same setting as in the report: episodes of 50 steps, random start, V = 0 at the beginning,
# step alpha = a / (n + b)^w where n = number of visits of the state.
#
# Three experiments:
#   1) "phase"   : alpha_n = c / (n + c - 1) for many c, gamma = 0.8. Stochastic approximation says
#                  the error should go like n^(-min(c(1-gamma), 1/2)), we check the slopes.
#   2) "horizon" : same rules but gamma from 0.5 to 0.98, how many updates to get 5% relative error.
#   3) "omega"   : alpha_n = 1/n^w, which w is best for a fixed budget, and does it depend on gamma.
#
# python step_sizes.py   (~3 min)

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import ex31_robot as robot  # noqa: E402

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SEEDS = 30
EP_LEN = 50
POL = robot.POLITIQUES["s/re"]

# transitions of s/re as small tables (2 outcomes max per state)
P1 = np.ones(2)
NEXT = np.zeros((2, 2), dtype=int)
REW = np.zeros((2, 2))
for s in (0, 1):
    outs = robot.TRANSITIONS[(s, POL[s])]
    P1[s] = outs[0][0]
    for k, (p, s2, r) in enumerate(outs):
        NEXT[s, k], REW[s, k] = s2, r
    if len(outs) == 1:
        NEXT[s, 1], REW[s, 1] = NEXT[s, 0], REW[s, 0]


def exact_v(gamma):
    P = np.zeros((2, 2))
    R = np.zeros(2)
    for s in (0, 1):
        for p, s2, r in robot.TRANSITIONS[(s, POL[s])]:
            P[s, s2] += p
            R[s] += p * r
    return np.linalg.solve(np.eye(2) - gamma * P, R)


def td_many(gammas, a, b, w, T, seed, n_grid=200):
    """One run per entry of the parameter arrays. Returns grid and relative error[time, run]."""
    rng = np.random.default_rng(seed)
    gammas, a, b, w = map(np.asarray, (gammas, a, b, w))
    nr = len(a)
    v = np.array([exact_v(g) for g in gammas])  # (nr, 2)
    scale = np.abs(v).max(axis=1)
    rows = np.arange(nr)
    V = np.zeros((nr, 2))
    n = np.zeros((nr, 2))
    grid = np.unique(np.logspace(1, np.log10(T), n_grid).astype(int))
    err = np.zeros((len(grid), nr))
    j = 0
    for t in range(1, T + 1):
        if (t - 1) % EP_LEN == 0:
            s = rng.integers(2, size=nr)
        k = (rng.random(nr) >= P1[s]).astype(int)
        s2, r = NEXT[s, k], REW[s, k]
        n[rows, s] += 1
        alpha = np.minimum(1.0, a / (n[rows, s] + b) ** w)
        V[rows, s] += alpha * (r + gammas * V[rows, s2] - V[rows, s])
        s = s2
        if j < len(grid) and t == grid[j]:
            err[j] = np.abs(V - v).max(axis=1) / scale
            j += 1
    return grid, err


def slope(grid, y, lo, hi):
    m = (grid >= lo) & (grid <= hi)
    return float(np.polyfit(np.log(grid[m]), np.log(y[m]), 1)[0])


def phase(T=1_000_000):
    gamma = 0.8
    cs = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10, 20]
    a = np.repeat(cs, SEEDS).astype(float)
    grid, err = td_many(np.full(len(a), gamma), a, a - 1, np.ones(len(a)), T, seed=7)
    out = {"gamma": gamma, "T": T, "seeds": SEEDS, "c": {}}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(cs):
        e = err[:, i * SEEDS:(i + 1) * SEEDS]
        m = e.mean(axis=1)
        sl = slope(grid, m, 1e4, T)
        pred = -min(c * (1 - gamma), 0.5)
        out["c"][str(c)] = {"slope_1e4_1e6": sl, "predicted": pred,
                            "final_rel_error_mean": float(e[-1].mean()),
                            "final_rel_error_sd": float(e[-1].std(ddof=1))}
        ax[0].plot(grid, m, color=cmap(i / (len(cs) - 1)), label=f"c = {c}")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("updates")
    ax[0].set_ylabel("relative error (mean of 30 seeds)")
    ax[0].legend(fontsize=7, ncol=2)
    cc = np.linspace(0.3, 20, 400)
    ax[1].plot(cc, [-min(x * (1 - gamma), 0.5) for x in cc], "k--", lw=1, label="-min(c(1-gamma), 1/2)")
    ax[1].plot(cs, [out["c"][str(c)]["slope_1e4_1e6"] for c in cs], "o", color="tab:red", label="measured slope")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("c in alpha_n = c/(n + c - 1)")
    ax[1].set_ylabel("slope of log error vs log n (1e4 to 1e6)")
    ax[1].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "steps_phase.png"), dpi=200)
    plt.close(fig)
    return out


def horizon(T=1_000_000, target=0.05):
    gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    rules = ["1/n", "H/(n+H-1)", "1/n^0.8", "1/n^0.6"]
    G, A, B, W, lab = [], [], [], [], []
    for g in gammas:
        H = 1 / (1 - g)
        for rname in rules:
            pa, pb, pw = {"1/n": (1, 0, 1), "H/(n+H-1)": (H, H - 1, 1),
                          "1/n^0.8": (1, 0, 0.8), "1/n^0.6": (1, 0, 0.6)}[rname]
            G += [g] * SEEDS
            A += [pa] * SEEDS
            B += [pb] * SEEDS
            W += [pw] * SEEDS
            lab.append((g, rname))
    grid, err = td_many(G, A, B, W, T, seed=8)
    out = {"T": T, "target": target, "seeds": SEEDS, "res": {}}
    for i, (g, rname) in enumerate(lab):
        e = err[:, i * SEEDS:(i + 1) * SEEDS].mean(axis=1)
        below = np.flatnonzero(e < target)
        # first time the mean curve goes under the target and stays under after
        first = None
        if len(below):
            ok = e < target
            stay = np.flatnonzero(np.cumprod(ok[::-1])[::-1])
            first = int(grid[stay[0]]) if len(stay) else None
        out["res"][f"{g}|{rname}"] = {"gamma": g, "rule": rname, "updates_to_target": first,
                                      "final_rel_error": float(e[-1])}
    # what the one-state calculation predicts for 1/n
    out["pred_1_over_n"] = {str(g): float((target * math.gamma(g)) ** (-1 / (1 - g))) for g in gammas}
    out["gammas"], out["rules"] = gammas, rules
    plot_horizon(out)
    return out


def plot_horizon(out):
    gammas, rules, T = out["gammas"], out["rules"], out["T"]
    plt.figure(figsize=(6.4, 4.2))
    Hs = [1 / (1 - g) for g in gammas]
    col = {"1/n": "tab:red", "H/(n+H-1)": "tab:orange", "1/n^0.8": "tab:blue", "1/n^0.6": "tab:purple"}
    for rname in rules:
        xs, ys = [], []
        for g, H in zip(gammas, Hs):
            u = out["res"][f"{g}|{rname}"]["updates_to_target"]
            if u is not None:
                xs.append(H)
                ys.append(u)
        plt.plot(xs, ys, "o-", color=col[rname], label=rname)
    pred = [out["pred_1_over_n"][str(g)] for g in gammas]
    plt.plot(Hs, pred, "k:", lw=1, label="1/n, one-state prediction")
    plt.axhline(T, color="grey", lw=0.8)
    plt.text(Hs[0], T * 1.3, "budget (1e6 updates)", fontsize=7, color="grey")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(Hs, [f"{h:g}" for h in np.round(Hs, 1)])
    plt.minorticks_off()
    plt.ylim(10, 1e9)
    plt.xlabel("effective horizon H = 1/(1 - gamma)")
    plt.ylabel(f"updates to reach {int(out['target'] * 100)}% relative error")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "steps_horizon.png"), dpi=200)
    plt.close()


def omega(T=100_000):
    gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    ws = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    G, W = [], []
    for g in gammas:
        for w in ws:
            G += [g] * SEEDS
            W += [w] * SEEDS
    grid, err = td_many(G, np.ones(len(G)), np.zeros(len(G)), W, T, seed=9)
    tail = grid >= 0.5 * T  # average the error on the last part, less noisy than the very last point
    M = np.zeros((len(gammas), len(ws)))
    for i in range(len(gammas)):
        for k in range(len(ws)):
            idx = (i * len(ws) + k)
            M[i, k] = err[tail][:, idx * SEEDS:(idx + 1) * SEEDS].mean()
    best = [ws[int(np.argmin(M[i]))] for i in range(len(gammas))]
    out = {"T": T, "gammas": gammas, "omegas": ws, "rel_error": M.tolist(), "best_omega": best}

    plt.figure(figsize=(6.4, 4.2))
    im = plt.imshow(np.log10(M), aspect="auto", cmap="magma_r", origin="lower")
    for i in range(len(gammas)):
        for k in range(len(ws)):
            plt.text(k, i, f"{M[i, k]:.3f}", ha="center", va="center", fontsize=7,
                     color="white" if np.log10(M[i, k]) > np.log10(M).mean() else "black")
        k_best = ws.index(best[i])
        plt.gca().add_patch(plt.Rectangle((k_best - 0.5, i - 0.5), 1, 1, fill=False, ec="cyan", lw=2))
    plt.colorbar(im, label="log10 relative error")
    plt.xticks(range(len(ws)), ws)
    plt.yticks(range(len(gammas)), gammas)
    plt.xlabel("w in alpha_n = 1/n^w")
    plt.ylabel("gamma")
    plt.title(f"relative error after {T} updates (best w boxed)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "steps_omega.png"), dpi=200)
    plt.close()
    return out


if __name__ == "__main__":
    # each part is saved on its own, so you can rerun only one: python step_sizes.py horizon
    parts = sys.argv[1:] or ["phase", "horizon", "omega"]
    path = os.path.join(RES, "step_sizes.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if parts == ["plot"]:  # only redraw the horizon figure from the saved numbers
        plot_horizon(res["horizon"])
        sys.exit()
    for part in parts:
        res[part] = {"phase": phase, "horizon": horizon, "omega": omega}[part]()
        print(part, "done", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)

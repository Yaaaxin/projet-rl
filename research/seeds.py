# Section 5 of the note: how much can we trust one seed?
#
# We rerun exactly the experiments of the report (exercises 3.1 and 3.2) on 100 seeds,
# using the same functions as extensions/graines.py (which itself calls the project code).
# Then for every sentence of the report that is a "claim" we count on how many seeds
# it would still be true. And we look at how many seeds you need before the ranking
# of two settings stops flipping.
#
# python seeds.py          -> runs the 100 seeds (~5 min on 2 cores) and does the analysis
# python seeds.py analyse  -> only the analysis, from results/seeds_raw.json

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "extensions"))
import graines  # noqa: E402
import ex31_robot as robot  # noqa: E402  (graines already put code/ in the path)
import ex32_etudiant as etu  # noqa: E402

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SEEDS = list(range(1, 101))
STEPS = ["0.5", "0.1", "0.01", "1/n"]


def both(seed):
    return {"seed": seed, "robot": graines.une_graine_robot(seed), "student": graines.une_graine_etudiant(seed)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float(x[n // 4: n - n // 4].mean())


def claims(raw):
    """Each claim is a function seed_result -> bool. The text is (more or less) what we wrote in the report."""
    v = {nom: robot.valeur_exacte(p) for nom, p in robot.POLITIQUES.items()}
    q_star_2R = 15.29

    def best_step(td, key=None):
        vals = {e: (td[e][key] if key else td[e]["erreur"]) for e in STEPS}
        return min(vals, key=vals.get)

    def mc_inside(r, rounded):
        for nom in robot.POLITIQUES:
            for s in (0, 1):
                est, ic = r["mc"][nom][s]
                tol = 0.005 if rounded else 0.0
                if abs(est - v[nom][s]) > ic + tol:
                    return False
        return True

    C = [
        ("R1", "robot TD: eta = 0.01 has the smallest error for s/w", lambda x: best_step(x["robot"]["td0"], "s/w") == "0.01"),
        ("R2", "robot TD: eta = 0.01 has the smallest error for s/s", lambda x: best_step(x["robot"]["td0"], "s/s") == "0.01"),
        ("R3", "robot TD: eta = 0.01 has the smallest error for s/re", lambda x: best_step(x["robot"]["td0"], "s/re") == "0.01"),
        ("R4", "robot TD: with 1/n the error for s/re is still above 1", lambda x: x["robot"]["td0"]["1/n"]["s/re"] > 1.0),
        ("R5", "robot TD: s/s is the hardest policy for every constant step",
         lambda x: all(x["robot"]["td0"][e]["s/s"] > max(x["robot"]["td0"][e]["s/w"], x["robot"]["td0"][e]["s/re"])
                       for e in ["0.5", "0.1", "0.01"])),
        ("R6", "robot MC: all true values inside the 95% intervals (as printed, 2 decimals)", lambda x: mc_inside(x["robot"], True)),
        ("R6s", "robot MC: same, but strict (no rounding)", lambda x: mc_inside(x["robot"], False)),
        ("R7", "robot MC eps-greedy with the real return finds s/re", lambda x: x["robot"]["mc_eps"]["politique"] == "search/recharge"),
        ("R8", "robot MC eps-greedy with the immediate reward finds s/w", lambda x: x["robot"]["mc_eps_immediat"] == "search/wait"),
        ("S1", "student TD: alpha = 0.01 has the smallest error", lambda x: best_step(x["student"]["td0"]) == "0.01"),
        ("S2", "student TD: 1/n is worse than 0.01", lambda x: x["student"]["td0"]["1/n"]["erreur"] > x["student"]["td0"]["0.01"]["erreur"]),
        ("S3", "student Q-learning: the three alphas all end on W/W/W/R",
         lambda x: all(x["student"]["q"][a]["bonne_a_la_fin"] for a in ["0.5", "0.1", "0.01"])),
        ("S4", "student Q-learning: alpha = 0.5 ends on W/W/W/R", lambda x: x["student"]["q"]["0.5"]["bonne_a_la_fin"]),
        ("S5", "student Q-learning: alpha = 0.01 becomes stable after 2500 to 4500 episodes",
         lambda x: x["student"]["q"]["0.01"]["episode_stable"] is not None and 2500 <= x["student"]["q"]["0.01"]["episode_stable"] <= 4500),
        ("S6", "student Q-learning: Q(2,R) with alpha = 0.01 is at least 3 below q*",
         lambda x: x["student"]["q"]["0.01"]["Q_2R"] < q_star_2R - 3),
    ]
    out = []
    for cid, text, f in C:
        k = sum(bool(f(x)) for x in raw)
        lo, hi = wilson(k, len(raw))
        out.append({"id": cid, "claim": text, "true_on": k, "n": len(raw), "ci95": [lo, hi]})
    return out


def comparisons(raw):
    """pairs (name, x values, y values) where we claimed x < y (lower is better)"""
    rob = [x["robot"]["td0"] for x in raw]
    stu = [x["student"] for x in raw]
    return [
        ("robot s/w: 0.01 vs 0.1", [r["0.01"]["s/w"] for r in rob], [r["0.1"]["s/w"] for r in rob]),
        ("robot s/s: 0.01 vs 1/n", [r["0.01"]["s/s"] for r in rob], [r["1/n"]["s/s"] for r in rob]),
        ("robot s/re: 0.01 vs 0.1", [r["0.01"]["s/re"] for r in rob], [r["0.1"]["s/re"] for r in rob]),
        ("student TD: 0.01 vs 1/n", [s["td0"]["0.01"]["erreur"] for s in stu], [s["td0"]["1/n"]["erreur"] for s in stu]),
        ("student Q: alpha 0.1 vs 0.5 (share of bad episodes)",
         [1 - s["q"]["0.1"]["part_bonne_5000_derniers"] for s in stu],
         [1 - s["q"]["0.5"]["part_bonne_5000_derniers"] for s in stu]),
    ]


def analyse():
    with open(os.path.join(RES, "seeds_raw.json")) as f:
        raw = json.load(f)
    out = {"n_seeds": len(raw), "claims": claims(raw)}

    # where were the seeds of the report (31 and 32) in the distribution?
    with open(os.path.join(HERE, "..", "resultats", "ex31.json")) as f:
        r31 = json.load(f)
    with open(os.path.join(HERE, "..", "resultats", "ex32.json")) as f:
        r32 = json.load(f)
    pct = {}
    for e in STEPS:
        for nom in robot.POLITIQUES:
            xs = [x["robot"]["td0"][e][nom] for x in raw]
            pct[f"robot {nom} {e}"] = float(np.mean(np.array(xs) < r31["td0"][e][nom]["erreur_moy_500_derniers"]) * 100)
        xs = [x["student"]["td0"][e]["erreur"] for x in raw]
        pct[f"student TD {e}"] = float(np.mean(np.array(xs) < r32["td0"][e]["erreur_moy_500_derniers"]) * 100)
    xs = [x["student"]["q"]["0.01"]["Q_2R"] for x in raw]
    pct["student Q(2,R) alpha 0.01"] = float(np.mean(np.array(xs) < r32["q_learning"]["0.01"]["Q"]["2,R"]) * 100)
    xs = [x["student"]["q"]["0.01"]["episode_stable"] for x in raw]
    pct["student stable episode alpha 0.01"] = float(np.mean(np.array(xs) < r32["q_learning"]["0.01"]["episode_stable"]) * 100)
    out["report_seed_percentile"] = pct

    # how many seeds before the ranking stops flipping (bootstrap over our 100 seeds)
    rng = np.random.default_rng(5)
    ks = [1, 2, 3, 5, 10, 20]
    comp = {}
    plt.figure(figsize=(6.4, 4.2))
    for name, xs, ys in comparisons(raw):
        xs, ys = np.array(xs), np.array(ys)
        ref = np.sign(ys.mean() - xs.mean())
        # probability of improvement: random x seed vs random y seed (Agarwal et al. 2021)
        poi = float(np.mean(xs[:, None] < ys[None, :]))
        agree = []
        for k in ks:
            ix = rng.integers(len(xs), size=(5000, k))
            iy = rng.integers(len(ys), size=(5000, k))
            agree.append(float(np.mean(np.sign(ys[iy].mean(axis=1) - xs[ix].mean(axis=1)) == ref)))
        comp[name] = {"mean_x": float(xs.mean()), "mean_y": float(ys.mean()), "median_x": float(np.median(xs)),
                      "median_y": float(np.median(ys)), "prob_improvement": poi,
                      "agreement_with_100_seeds": dict(zip(map(str, ks), agree))}
        plt.plot(ks, agree, "o-", label=f"{name} (P(x<y) = {poi:.2f})")
    plt.xscale("log")
    plt.xticks(ks, ks)
    plt.ylim(0.45, 1.02)
    plt.xlabel("number of seeds averaged")
    plt.ylabel("P(same ranking as with 100 seeds)")
    plt.legend(fontsize=6.5)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "seeds_agreement.png"), dpi=200)
    plt.close()
    out["comparisons"] = comp

    # shapes of a few distributions, mean vs median vs IQM
    shapes = {
        "robot s/s, step 1/n (TD error)": [x["robot"]["td0"]["1/n"]["s/s"] for x in raw],
        "student, step 1/n (TD error)": [x["student"]["td0"]["1/n"]["erreur"] for x in raw],
        "student Q, alpha 0.5 (share of good episodes, last 5000)": [x["student"]["q"]["0.5"]["part_bonne_5000_derniers"] for x in raw],
        "student Q, alpha 0.01 (episode where policy becomes stable)": [x["student"]["q"]["0.01"]["episode_stable"] for x in raw],
    }
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    sh = {}
    for a, (name, xs) in zip(ax, shapes.items()):
        xs = np.array(xs, dtype=float)
        a.hist(xs, bins=20, color="tab:blue", alpha=0.75, edgecolor="k")
        a.axvline(xs.mean(), color="tab:red", lw=1.5, label="mean")
        a.axvline(np.median(xs), color="k", ls="--", lw=1.2, label="median")
        a.set_title(name, fontsize=8)
        a.grid(alpha=0.3)
        z = (xs - xs.mean()) / xs.std()
        sh[name] = {"mean": float(xs.mean()), "median": float(np.median(xs)), "iqm": iqm(xs),
                    "sd": float(xs.std(ddof=1)), "min": float(xs.min()), "max": float(xs.max()),
                    "skew": float(np.mean(z ** 3))}
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "seeds_shapes.png"), dpi=200)
    plt.close(fig)
    out["shapes"] = sh

    with open(os.path.join(RES, "seeds.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] != "analyse":
        with Pool(2) as pool:
            raw = pool.map(both, SEEDS)
        with open(os.path.join(RES, "seeds_raw.json"), "w") as f:
            json.dump(raw, f)
    analyse()

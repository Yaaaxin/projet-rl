# Dans le rapport on a dit : "pour 3.1 et 3.2 on a utilise qu'une graine, il faudrait repeter".
# Bon, on le fait ici. On relance EXACTEMENT les memes experiences que dans le projet
# (memes fonctions, memes parametres, meme ordre), mais avec 30 graines differentes.
#
# Petit test au debut : avec la graine du projet (31 et 32) on doit retomber sur les memes
# chiffres que dans resultats/ex31.json et ex32.json, sinon c'est qu'on a pas refait pareil.
#
# Lancer :  python graines.py   (environ 5 min sur 2 coeurs)

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "code"))
import ex31_robot as robot      # noqa: E402
import ex32_etudiant as etu     # noqa: E402

DOSSIER_FIG = os.path.join(ICI, "figures")
DOSSIER_RES = os.path.join(ICI, "resultats")
os.makedirs(DOSSIER_FIG, exist_ok=True)
os.makedirs(DOSSIER_RES, exist_ok=True)

GRAINES = list(range(1, 31))
PAS = (0.5, 0.1, 0.01, "1/n")


def une_graine_robot(graine):
    """meme chose que main() de ex31_robot.py, sans les figures"""
    rng = np.random.default_rng(graine)
    out = {"mc": {}, "td0": {}}
    for nom, p in robot.POLITIQUES.items():
        est, ic, _ = robot.monte_carlo(p, 2000, rng)
        out["mc"][nom] = [[float(est[s]), float(ic[s])] for s in (0, 1)]
    for eta in PAS:
        out["td0"][str(eta)] = {}
        for nom, p in robot.POLITIQUES.items():
            V, err = robot.td0(p, eta, 2000, rng, v_exacte=robot.valeur_exacte(p))
            out["td0"][str(eta)][nom] = float(err[-500:].mean())
    Q, glout, _ = robot.mc_eps_greedy(0.1, 3000, rng)
    out["mc_eps"] = {"politique": f"{glout[0]}/{glout[1]}",
                     "Q": {f"{robot.NOMS_ETATS[s]},{a}": float(v) for (s, a), v in Q.items()}}
    _, glout_i, _ = robot.mc_eps_greedy(0.1, 3000, rng, recompense_immediate=True)
    out["mc_eps_immediat"] = f"{glout_i[0]}/{glout_i[1]}"
    return out


def une_graine_etudiant(graine):
    """meme chose que main() de ex32_etudiant.py, sans les figures"""
    rng = np.random.default_rng(graine)
    v_swws = etu.valeur_exacte(etu.SWWS)
    pol_opt, _, _ = etu.iteration_politique({s: "W" for s in etu.NON_ABS})
    out = {"td0": {}, "q": {}}
    for alpha in PAS:
        V, err = etu.td0(etu.SWWS, alpha, 5000, rng, v_swws)
        out["td0"][str(alpha)] = {"erreur": float(err[-500:].mean()), "V": [float(V[s]) for s in etu.NON_ABS]}
    for alpha in (0.5, 0.1, 0.01):
        Q, glout, bon = etu.q_learning(alpha, 0.1, 20000, rng, pol_opt)
        stable = int(np.argmax(np.cumprod(bon[::-1])[::-1])) if bon[-1] else None
        out["q"][str(alpha)] = {"bonne_a_la_fin": bool(bon[-1]), "episode_stable": stable,
                                "part_bonne_5000_derniers": float(bon[-5000:].mean()),
                                "Q_2R": float(Q[(2, "R")])}
    return out


def verif_graine_projet():
    """La graine du projet doit redonner les chiffres du rapport."""
    with open(os.path.join(ICI, "..", "resultats", "ex31.json"), encoding="utf-8") as f:
        r31 = json.load(f)
    with open(os.path.join(ICI, "..", "resultats", "ex32.json"), encoding="utf-8") as f:
        r32 = json.load(f)
    a = une_graine_robot(robot.GRAINE)
    b = une_graine_etudiant(etu.GRAINE)
    ok = abs(a["td0"]["0.01"]["s/re"] - r31["td0"]["0.01"]["s/re"]["erreur_moy_500_derniers"]) < 1e-12
    ok &= a["mc_eps"]["politique"] == r31["mc_eps_greedy"]["politique"]
    ok &= abs(b["td0"]["0.01"]["erreur"] - r32["td0"]["0.01"]["erreur_moy_500_derniers"]) < 1e-12
    ok &= b["q"]["0.01"]["episode_stable"] == r32["q_learning"]["0.01"]["episode_stable"]
    return ok, a, b


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"moy": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()),
            "q25": float(np.percentile(x, 25)), "med": float(np.median(x)),
            "q75": float(np.percentile(x, 75)), "max": float(x.max())}


def main():
    ok, projet31, projet32 = verif_graine_projet()
    print("graine du projet retrouvee :", ok)
    with Pool(2) as pool:
        rob = pool.map(une_graine_robot, GRAINES)
        etd = pool.map(une_graine_etudiant, GRAINES)

    res = {"graines": GRAINES, "verif_graine_projet": bool(ok), "robot": {}, "etudiant": {}}

    # robot : couverture des intervalles de confiance de Monte-Carlo
    couverture = {}
    for nom, p in robot.POLITIQUES.items():
        v = robot.valeur_exacte(p)
        for s in (0, 1):
            dedans = [abs(r["mc"][nom][s][0] - v[s]) <= r["mc"][nom][s][1] for r in rob]
            couverture[f"{nom},{robot.NOMS_ETATS[s]}"] = float(np.mean(dedans))
    res["robot"]["couverture_ic95_mc"] = couverture
    res["robot"]["td0_erreur"] = {str(e): {nom: stats([r["td0"][str(e)][nom] for r in rob])
                                           for nom in robot.POLITIQUES} for e in PAS}
    res["robot"]["td0_erreur_projet"] = projet31["td0"]
    res["robot"]["mc_eps_politiques"] = {p: sum(r["mc_eps"]["politique"] == p for r in rob)
                                         for p in sorted({r["mc_eps"]["politique"] for r in rob})}
    res["robot"]["mc_eps_immediat_politiques"] = {p: sum(r["mc_eps_immediat"] == p for r in rob)
                                                  for p in sorted({r["mc_eps_immediat"] for r in rob})}
    res["robot"]["mc_eps_Q"] = {k: stats([r["mc_eps"]["Q"][k] for r in rob]) for k in rob[0]["mc_eps"]["Q"]}

    # etudiant
    res["etudiant"]["td0_erreur"] = {str(a): stats([r["td0"][str(a)]["erreur"] for r in etd]) for a in PAS}
    res["etudiant"]["td0_erreur_projet"] = {str(a): projet32["td0"][str(a)]["erreur"] for a in PAS}
    res["etudiant"]["q"] = {}
    for a in ("0.5", "0.1", "0.01"):
        stables = [r["q"][a]["episode_stable"] for r in etd]
        res["etudiant"]["q"][a] = {
            "nb_graines_bonne_a_la_fin": int(sum(r["q"][a]["bonne_a_la_fin"] for r in etd)),
            "episode_stable": stats([x for x in stables if x is not None]),
            "part_bonne_5000_derniers": stats([r["q"][a]["part_bonne_5000_derniers"] for r in etd]),
            "Q_2R": stats([r["q"][a]["Q_2R"] for r in etd]),
        }
    res["etudiant"]["q_projet"] = projet32["q"]

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    pos = 0
    ticks, labels = [], []
    for e in PAS:
        for k, nom in enumerate(robot.POLITIQUES):
            x = [r["td0"][str(e)][nom] for r in rob]
            axes[0].boxplot([x], positions=[pos], widths=0.6)
            axes[0].plot(pos, projet31["td0"][str(e)][nom], "r*", ms=9)
            ticks.append(pos)
            labels.append(f"{nom}\n{e}")
            pos += 1
        pos += 0.8
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_title("Robot, TD(0) : erreur moyenne des 500 derniers episodes\n(politique / pas, etoile = graine du projet)",
                      fontsize=10)
    axes[0].grid(alpha=0.3, which="both")

    x = [[r["td0"][str(a)]["erreur"] for r in etd] for a in PAS]
    axes[1].boxplot(x, tick_labels=[str(a) for a in PAS])
    axes[1].plot(range(1, 5), [projet32["td0"][str(a)]["erreur"] for a in PAS], "r*", ms=9)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("alpha")
    axes[1].set_title("Etudiant, TD(0) pour SWWS : meme chose", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIG, "graines_td0.png"), dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 4.2))
    st = [r["q"]["0.01"]["episode_stable"] for r in etd]
    plt.hist(st, bins=12, color="tab:green", alpha=0.7, edgecolor="k")
    plt.axvline(projet32["q"]["0.01"]["episode_stable"], color="r", ls="--",
                label=f"graine du projet ({projet32['q']['0.01']['episode_stable']})")
    plt.xlabel("episode a partir duquel la politique gloutonne reste W/W/W/R")
    plt.ylabel("nombre de graines")
    plt.title("Q-learning, alpha = 0.01 : sur 30 graines")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "graines_qlearning.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "graines.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "graines"}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()

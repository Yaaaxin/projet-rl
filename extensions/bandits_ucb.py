# Suite de l'exercice 2.4 : est-ce qu'on peut faire mieux que ETC et eps_n-greedy ?
# On compare avec UCB, KL-UCB et Thompson sampling sur un horizon beaucoup plus long
# (n = 100 000) et on regarde la borne de Lai-Robbins.
#
# Pour aller vite on simule toutes les repetitions en meme temps (tableaux numpy
# de taille (R, K)), sinon avec des boucles python ça prend des heures.
#
# Lancer :  python bandits_ucb.py   (environ 2-3 min)

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_FIG = os.path.join(ICI, "figures")
DOSSIER_RES = os.path.join(ICI, "resultats")
os.makedirs(DOSSIER_FIG, exist_ok=True)
os.makedirs(DOSSIER_RES, exist_ok=True)

LAMBDAS = np.array([2.0, 3.0, 4.0])
K = 3
DELTA = LAMBDAS.max() - LAMBDAS  # (2, 1, 0)
N = 100_000
R = 200  # nombre de repetitions
GRAINE = 24
# une couleur par algo, la meme sur toutes les figures
COULEURS = {"ETC(10)": "tab:blue", "eps_n-greedy (c=1)": "tab:orange", "UCB (sigma=1)": "tab:green",
            "UCB (sigma=2)": "tab:red", "KL-UCB": "tab:purple", "Thompson": "tab:brown"}


def kl_poisson(p, q):
    # KL(P(p) || P(q)) = p log(p/q) + q - p, avec la convention 0 log 0 = 0
    p = np.maximum(p, 1e-12)
    return p * np.log(p / q) + q - p


# constante de Lai-Robbins : sum_k Delta(k) / KL(mu_k, mu*)  (on enleve la machine optimale)
C_LR = sum(DELTA[k] / kl_poisson(LAMBDAS[k], LAMBDAS.max()) for k in range(K) if DELTA[k] > 0)


def indice_kl_ucb(mu_hat, nb, t, n_iter=22):
    """plus grand q >= mu_hat tel que nb * KL(mu_hat, q) <= log t (bisection)."""
    seuil = np.log(t) / nb
    bas = mu_hat.copy()
    haut = mu_hat + 1.0
    # on double jusqu'a depasser le seuil (en pratique 2-3 tours max)
    for _ in range(40):
        trop_bas = kl_poisson(mu_hat, haut) < seuil
        if not trop_bas.any():
            break
        haut = np.where(trop_bas, mu_hat + 2 * (haut - mu_hat), haut)
    for _ in range(n_iter):
        milieu = (bas + haut) / 2
        ok = kl_poisson(mu_hat, milieu) <= seuil
        bas = np.where(ok, milieu, bas)
        haut = np.where(ok, haut, milieu)
    return bas


def argmax_hasard(x, rng):
    # argmax par ligne, en cassant les egalites au hasard (petit bruit)
    return np.argmax(x + 1e-9 * rng.random(x.shape), axis=1)


def simuler(nom, rng, n=N, r=R):
    """Renvoie le regret cumule moyen sur une grille log + le regret final de chaque run."""
    somme = np.zeros((r, K))
    nb = np.zeros((r, K))
    regret = np.zeros(r)
    lignes = np.arange(r)
    grille = np.unique(np.logspace(0, np.log10(n), 300).astype(int))
    courbe = []
    j = 0

    for t in range(1, n + 1):
        if nom.startswith("ETC"):
            m = 10
            if t <= m * K:
                a = np.full(r, (t - 1) % K)
            else:
                a = argmax_hasard(somme / nb, rng)
        elif nom.startswith("eps"):
            eps = min(1.0, 1.0 * K / t)  # c = 1, delta = 1 comme dans le projet
            mu_hat = np.divide(somme, nb, out=np.zeros((r, K)), where=nb > 0)
            a = argmax_hasard(mu_hat, rng)
            explore = rng.random(r) < eps
            a = np.where(explore, rng.integers(K, size=r), a)
        elif t <= K:
            a = np.full(r, t - 1)  # les autres : on joue chaque machine une fois au debut
        else:
            mu_hat = somme / nb
            if nom == "UCB (sigma=1)":
                indice = mu_hat + np.sqrt(2 * np.log(t) / nb)
            elif nom == "UCB (sigma=2)":
                indice = mu_hat + np.sqrt(2 * 4 * np.log(t) / nb)
            elif nom == "KL-UCB":
                indice = indice_kl_ucb(mu_hat, nb, t)
            elif nom == "Thompson":
                # loi a posteriori Gamma(0.5 + somme, nb) (prior de Jeffreys)
                indice = rng.gamma(0.5 + somme, 1.0 / nb)
            a = argmax_hasard(indice, rng)

        x = rng.poisson(LAMBDAS[a])
        somme[lignes, a] += x
        nb[lignes, a] += 1
        regret += DELTA[a]
        if j < len(grille) and t == grille[j]:
            courbe.append(regret.mean())
            j += 1
        if t == 1000:
            regret_1000 = regret.copy()  # pour comparer avec le projet (n = 1000)
    return grille, np.array(courbe), regret.copy(), regret_1000


def main():
    noms = ["ETC(10)", "eps_n-greedy (c=1)", "UCB (sigma=1)", "UCB (sigma=2)", "KL-UCB", "Thompson"]
    res = {"C_lai_robbins": float(C_LR), "N": N, "R": R}
    courbes = {}
    finals = {}
    for i, nom in enumerate(noms):
        rng = np.random.default_rng(GRAINE + i)
        debut = time.time()
        grille, courbe, final, r1000 = simuler(nom, rng)
        courbes[nom] = courbe
        finals[nom] = final
        # un run "bloque" = regret final enorme, on prend le seuil 0.1 * N
        res[nom] = {
            "regret_moyen_n1000": float(r1000.mean()),
            "regret_moyen": float(final.mean()),
            "regret_ecart_type": float(final.std(ddof=1)),
            "regret_median": float(np.median(final)),
            "regret_max": float(final.max()),
            "part_runs_bloques": float(np.mean(final > 0.1 * N)),
            "regret_sur_log_n": float(final.mean() / np.log(N)),
        }
        print(f"{nom:22s} regret moyen {final.mean():9.1f}  median {np.median(final):8.1f}"
              f"  bloques {np.mean(final > 0.1 * N):.2f}   ({time.time() - debut:.0f} s)")

    # figure 1 : regret moyen, axe des x en log, avec la borne C log n
    plt.figure(figsize=(7.5, 4.6))
    for nom in noms:
        plt.plot(grille, courbes[nom], color=COULEURS[nom], label=nom)
    g = grille[grille >= 3]
    plt.plot(g, C_LR * np.log(g), "k--", lw=1.2, label=f"Lai-Robbins : {C_LR:.1f} log n")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1, None)
    plt.xlabel("n")
    plt.ylabel("regret moyen (200 runs)")
    plt.title("Bandits de Poisson : regret moyen jusqu'a n = 100 000")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "bandits_regret_log.png"), dpi=200)
    plt.close()

    # figure 2 : regret / log n pour les algos "intelligents"
    plt.figure(figsize=(7.5, 4.3))
    garde = grille >= 20
    for nom in ["UCB (sigma=1)", "UCB (sigma=2)", "KL-UCB", "Thompson"]:
        plt.plot(grille[garde], courbes[nom][garde] / np.log(grille[garde]), color=COULEURS[nom], label=nom)
    plt.axhline(C_LR, color="k", ls="--", lw=1.2, label=f"constante de Lai-Robbins ({C_LR:.2f})")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("regret moyen / log n")
    plt.title("Est-ce qu'on s'approche de la borne ?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "bandits_regret_sur_log.png"), dpi=200)
    plt.close()

    # figure 3 : distribution du regret final (c'est la que se voient les runs bloques)
    plt.figure(figsize=(7.5, 4.3))
    plt.boxplot([np.maximum(finals[nom], 1) for nom in noms], tick_labels=noms)
    plt.yscale("log")
    plt.xticks(rotation=15, fontsize=8)
    plt.ylabel("regret final a n = 100 000 (log)")
    plt.title("Regret final de chaque run")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "bandits_regret_final.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "bandits_ucb.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

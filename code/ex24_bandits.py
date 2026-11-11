# -*- coding: utf-8 -*-
"""
Exercice 2.4 : bandits avec des lois de Poisson.

Trois machines : P(2), P(3), P(4). On compare ETC(m) pour m = 3, 5, 10
et la politique eps_n-greedy (c = 1), avec 100 essais de 1000 tirages.

Lancer :  python ex24_bandits.py
Les figures vont dans ../figures et les chiffres dans ../resultats/ex24.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas besoin d'afficher, on enregistre juste les images
import matplotlib.pyplot as plt

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_FIG = os.path.join(ICI, "..", "figures")
DOSSIER_RES = os.path.join(ICI, "..", "resultats")
os.makedirs(DOSSIER_FIG, exist_ok=True)
os.makedirs(DOSSIER_RES, exist_ok=True)

# donnees du probleme
LAMBDAS = np.array([2.0, 3.0, 4.0])  # moyennes mu(k) des trois machines
K = len(LAMBDAS)
MU_STAR = LAMBDAS.max()  # = 4, c'est la machine 3
DELTA = MU_STAR - LAMBDAS  # gaps : (2, 1, 0)
N_TIRAGES = 1000  # horizon n
N_ESSAIS = 100  # nombre de repetition
GRAINE = 2026  # comme ça on retombe sur les memes chiffres


def argmax_aleatoire(valeurs, rng):
    """argmax, mais en cas d'égalité on tire au sort parmi les meilleurs
    (c'est le « une des machines » du poly)."""
    meilleurs = np.flatnonzero(valeurs == valeurs.max())
    return rng.choice(meilleurs)


# ETC(m), def 4.5 du chap 2
def etc(m, rng, n=N_TIRAGES, recompenses_forcees=None):
    """Joue ETC(m) pendant n tirages.

    Exploration : on joue 1, 2, 3, 1, 2, 3, ... (m fois chaque machine).
    Ensuite : A_i dans argmax mu_hat_{i-1}. Attention dans le poly les
    moyennes continuent d'être mise à jour après l'exploration (c'est
    une politique « greedy » à partir de mK + 1).

    recompenses_forcees : liste optionnelle qui remplace les premières
    récompenses (ça sert juste a tester l'evenement de la question 2a).
    """
    somme = np.zeros(K)  # somme des récompenses de chaque machine
    nb = np.zeros(K)  # N_n(k) : nombre de fois où on a joué k
    actions = np.empty(n, dtype=int)
    recompenses = np.empty(n)

    for i in range(n):
        if i < m * K:
            a = i % K  # exploration : 1, 2, 3, 1, 2, 3...
        else:
            mu_hat = somme / nb  # ici nb > 0 partout, pas de division par 0
            a = argmax_aleatoire(mu_hat, rng)

        if recompenses_forcees is not None and i < len(recompenses_forcees):
            x = recompenses_forcees[i]
        else:
            x = rng.poisson(LAMBDAS[a])

        # on met a jour seulement la machine qu'on a joué
        somme[a] += x
        nb[a] += 1
        actions[i] = a
        recompenses[i] = x
    return actions, recompenses


# eps_n-greedy (def 4.9 chap 2)
def eps_n_greedy(c, rng, n=N_TIRAGES, delta=1.0):
    """eps_n = min(1, cK / (delta^2 n)).

    Avec probabilité eps_n on explore (machine uniforme), sinon on joue
    une machine de argmax mu_hat. Une machine jamais jouée a mu_hat = 0
    (c'est une convention a nous, de toute façon on explore au tout debut).
    """
    somme = np.zeros(K)
    nb = np.zeros(K)
    actions = np.empty(n, dtype=int)
    recompenses = np.empty(n)

    for i in range(n):
        t = i + 1  # t commence a 1 sinon eps = c*K/0
        eps = min(1.0, c * K / (delta ** 2 * t))
        if rng.random() < eps:
            a = rng.integers(K)  # on explore
        else:
            mu_hat = np.divide(somme, nb, out=np.zeros(K), where=nb > 0)
            a = argmax_aleatoire(mu_hat, rng)  # on exploite
        x = rng.poisson(LAMBDAS[a])
        somme[a] += x
        nb[a] += 1
        actions[i] = a
        recompenses[i] = x
    return actions, recompenses


# stats sur les 100 essais
def statistiques(politique, rng, n_essais=N_ESSAIS, debut_exploitation=None):
    """Lance la politique n_essais fois et résume les résultats.

    debut_exploitation : pour ETC(m), l'indice mK du premier tirage « greedy ».
    On regarde si la machine choisie à ce moment-là est encore celle de la fin
    (sinon, ça veut dire que la politique s'est corrigée toute seule).
    """
    totaux = []  # récompense totale de chaque essai
    nb_corrections = 0  # nb d'essais ou la machine change apres l'exploration
    machine_finale = []  # machine jouée au dernier tirage
    part_bonne = []  # % des 100 derniers tirages sur la machine 3
    regret_cumule = np.zeros(N_TIRAGES)  # pseudo-regret moyen sum Delta(A_i)

    for _ in range(n_essais):
        actions, recompenses = politique(rng)
        totaux.append(recompenses.sum())
        machine_finale.append(actions[-1])
        part_bonne.append(np.mean(actions[-100:] == 2))
        if debut_exploitation is not None and actions[debut_exploitation] != actions[-1]:
            nb_corrections += 1
        regret_cumule += np.cumsum(DELTA[actions])
    regret_cumule /= n_essais

    totaux = np.array(totaux)
    machine_finale = np.array(machine_finale)
    return {
        "frequence_machine_finale": [float(np.mean(machine_finale == k)) for k in range(K)],
        "recompense_moyenne": float(totaux.mean()),
        "recompense_ecart_type": float(totaux.std(ddof=1)),
        "recompense_min": float(totaux.min()),
        "recompense_max": float(totaux.max()),
        "part_machine3_fin": float(np.mean(part_bonne)),
        "regret_final": float(regret_cumule[-1]),
        "nb_corrections": nb_corrections,
        "courbe_regret": regret_cumule,
        "totaux": totaux,
    }


def main():
    rng = np.random.default_rng(GRAINE)
    resultats = {}

    # Q2a : on verifie l'evenement qui bloque ETC(3) sur la machine 1
    # Pendant l'exploration (m = 3) : la machine 1 donne 1, 1, 1 et les
    # machines 2 et 3 donnent 0. Ensuite tout est vraiment aléatoire.
    forcees = [1, 0, 0] * 3
    actions, _ = etc(3, rng, recompenses_forcees=forcees)
    resultats["test_evenement"] = {
        "nb_fois_machine1_apres_exploration": int(np.sum(actions[9:] == 0)),
        "nb_tirages_apres_exploration": int(len(actions) - 9),
    }
    # proba exacte de cet evenement (P(X=1) = 2e^{-2} pour la machine 1)
    p_evt = (2 * np.exp(-2)) ** 3 * np.exp(-3) ** 3 * np.exp(-4) ** 3
    # evenement un peu plus grand : somme des 3 recompenses de la machine 1 > 0
    p_large = (1 - np.exp(-6)) * np.exp(-9) * np.exp(-12)
    resultats["proba_evenement"] = {"exact_111": float(p_evt), "large": float(p_large)}

    # Q2b et Q2c : ETC(m) pour m = 3, 5, 10
    courbes = {}
    boites = {}  # pour les boites a moustache
    for m in (3, 5, 10):
        stats = statistiques(lambda r, m=m: etc(m, r), rng, debut_exploitation=m * K)
        courbes[f"ETC({m})"] = stats.pop("courbe_regret")
        boites[f"ETC({m})"] = stats.pop("totaux")
        resultats[f"ETC({m})"] = stats

    # Q3 : eps_n-greedy avec c = 1, delta = 1
    for c in (1, 10):  # on a rajouté c = 10 pour comparer
        stats = statistiques(lambda r, c=c: eps_n_greedy(c, r), rng)
        courbes[f"eps_n-greedy (c={c})"] = stats.pop("courbe_regret")
        boites[f"eps_n (c={c})"] = stats.pop("totaux")
        stats.pop("nb_corrections")
        resultats[f"eps_n-greedy (c={c})"] = stats

    # figure 1 : regret moyen
    plt.figure(figsize=(7, 4.2))
    for nom, courbe in courbes.items():
        plt.plot(np.arange(1, N_TIRAGES + 1), courbe, label=nom)
    plt.xlabel("nombre de tirages n")
    plt.ylabel("regret moyen  $\\sum_k \\Delta(k)\\,N_n(k)$")
    plt.title("Exercice 2.4 : regret moyen sur 100 essais")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex24_regret.png"), dpi=200)
    plt.close()

    # boxplot des recompenses totales (pas dans le rapport finalement)
    plt.figure(figsize=(7, 4.2))
    plt.boxplot(list(boites.values()), tick_labels=list(boites.keys()))
    plt.axhline(MU_STAR * N_TIRAGES, color="grey", ls="--", lw=1,
                label="4000 = ce qu'on gagnerait en moyenne avec la machine 3")
    plt.ylabel("récompense totale sur 1000 tirages")
    plt.title("Exercice 2.4 : récompense totale (100 essais)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex24_totaux.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "ex24.json"), "w") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(json.dumps(resultats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

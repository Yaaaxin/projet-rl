# -*- coding: utf-8 -*-
"""
Exercice 3.2 : le dilemme de l'étudiant (Exemple 2.14 du chapitre 1), gamma = 0.8.

États 1..7, les états 5, 6, 7 sont absorbants. Actions : R (se reposer), W (travailler).
On lit les probabilités et les récompenses sur la figure du poly.
Convention (celle du cours) : la récompense r(s') est reçue en arrivant dans s',
et après un état absorbant tout vaut 0, donc v(5) = v(6) = v(7) = 0.

1. TD(0) pour la politique SWWS (R en 1, W en 2, W en 3, R en 4)
2. Q-learning pour retrouver la politique optimale

Lancer :  python ex32_etudiant.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_FIG = os.path.join(ICI, "..", "figures")
DOSSIER_RES = os.path.join(ICI, "..", "resultats")
os.makedirs(DOSSIER_FIG, exist_ok=True)
os.makedirs(DOSSIER_RES, exist_ok=True)

GAMMA = 0.8
ETATS = [1, 2, 3, 4, 5, 6, 7]
ABSORBANTS = {5, 6, 7}
ACTIONS = ["R", "W"]
# recompense qu'on recoit en arrivant dans l'etat (en rouge sur la figure)
r = {1: 0, 2: 1, 3: -1, 4: -10, 5: -10, 6: 100, 7: -1000}

# P[(s, a)] = {etat suivant : proba}  (en bleu sur la figure)
P = {
    (1, "R"): {1: 0.5, 2: 0.5},
    (1, "W"): {1: 0.5, 3: 0.5},
    (2, "R"): {2: 0.6, 5: 0.4},
    (2, "W"): {1: 0.3, 3: 0.7},
    (3, "R"): {2: 0.4, 3: 0.6},
    (3, "W"): {3: 0.5, 4: 0.5},
    (4, "R"): {6: 0.9, 4: 0.1},
    (4, "W"): {7: 1.0},
}
NON_ABS = [1, 2, 3, 4]
SWWS = {1: "R", 2: "W", 3: "W", 4: "R"}
GRAINE = 32


def r_barre(s, a):
    return sum(p * r[s2] for s2, p in P[(s, a)].items())


def valeur_exacte(politique):
    """v = R_Pi + gamma P_Pi v sur les états 1..4 (v = 0 sur les absorbants)."""
    idx = {s: i for i, s in enumerate(NON_ABS)}
    A = np.eye(4)
    b = np.zeros(4)
    for s in NON_ABS:
        a = politique[s]
        b[idx[s]] = r_barre(s, a)
        for s2, p in P[(s, a)].items():
            if s2 not in ABSORBANTS:
                A[idx[s], idx[s2]] -= GAMMA * p
    v = np.linalg.solve(A, b)
    return {s: float(v[idx[s]]) for s in NON_ABS}


def q_depuis_v(v):
    vv = dict(v)
    vv.update({s: 0.0 for s in ABSORBANTS})
    return {(s, a): r_barre(s, a) + GAMMA * sum(p * vv[s2] for s2, p in P[(s, a)].items())
            for s in NON_ABS for a in ACTIONS}


def iteration_politique(depart):
    """Pour avoir la vrai réponse (avec le modèle) et vérifier Q-learning."""
    pol = dict(depart)
    while True:
        v = valeur_exacte(pol)
        q = q_depuis_v(v)
        nouv = {}
        for s in NON_ABS:
            meilleure = max(ACTIONS, key=lambda a: q[(s, a)])
            nouv[s] = pol[s] if np.isclose(q[(s, meilleure)], q[(s, pol[s])]) else meilleure
        if nouv == pol:
            return pol, v, q
        pol = nouv


# simulateur : l'etudiant connait pas P, il voit juste ce qui lui arrive
def un_pas(s, a, rng):
    u = rng.random()
    cumul = 0.0
    for s2, p in P[(s, a)].items():
        cumul += p
        if u < cumul:
            return s2, r[s2]
    return s2, r[s2]


T_MAX = 200  # au cas ou un episode finit jamais (en pratique ça arrive quasi jamais)


# Q1 : TD(0) pour SWWS
def td0(politique, alpha, n_episodes, rng, v_exacte):
    """Chaque épisode part d'un état tiré au hasard dans {1, 2, 3, 4}
    et s'arrête quand on tombe dans un état absorbant."""
    V = {s: 0.0 for s in ETATS}  # V reste a 0 sur 5, 6, 7
    visites = {s: 0 for s in NON_ABS}
    erreurs = []
    for _ in range(n_episodes):
        s = int(rng.choice(NON_ABS))
        for _ in range(T_MAX):
            if s in ABSORBANTS:
                break
            s2, rec = un_pas(s, politique[s], rng)
            visites[s] += 1
            pas = 1.0 / visites[s] if alpha == "1/n" else alpha
            V[s] += pas * (rec + GAMMA * V[s2] - V[s])
            s = s2
        erreurs.append(max(abs(V[s] - v_exacte[s]) for s in NON_ABS))
    return V, np.array(erreurs)


# Q2 : Q-learning (algo 6 du chap 3)
def q_learning(alpha, eps, n_episodes, rng, politique_opt):
    """On explore avec eps-greedy, mais on met à jour avec max_a Q(S', a)."""
    Q = {(s, a): 0.0 for s in ETATS for a in ACTIONS}  # Q = 0 sur les etats absorbants
    bonne_politique = []  # 1 si la politique gloutonne est deja la bonne
    for _ in range(n_episodes):
        s = int(rng.choice(NON_ABS))
        for _ in range(T_MAX):
            if s in ABSORBANTS:
                break
            # eps-greedy (si egalité on tire au hasard)
            if rng.random() < eps:
                a = ACTIONS[rng.integers(2)]
            else:
                qs = [Q[(s, b)] for b in ACTIONS]
                a = ACTIONS[rng.integers(2)] if qs[0] == qs[1] else ACTIONS[int(np.argmax(qs))]
            s2, rec = un_pas(s, a, rng)
            cible = rec + GAMMA * max(Q[(s2, b)] for b in ACTIONS)
            Q[(s, a)] += alpha * (cible - Q[(s, a)])
            s = s2
        gloutonne = {x: max(ACTIONS, key=lambda b: Q[(x, b)]) for x in NON_ABS}
        bonne_politique.append(gloutonne == politique_opt)
    return Q, gloutonne, np.array(bonne_politique)


def main():
    rng = np.random.default_rng(GRAINE)
    res = {"r_barre": {f"{s},{a}": r_barre(s, a) for s in NON_ABS for a in ACTIONS}}

    v_swws = valeur_exacte(SWWS)
    pol_opt, v_opt, q_opt = iteration_politique({s: "W" for s in NON_ABS})
    res["v_SWWS_exacte"] = v_swws
    res["politique_optimale"] = pol_opt
    res["v_star"] = v_opt
    res["q_star"] = {f"{s},{a}": float(x) for (s, a), x in q_opt.items()}

    # Q1
    N_EP = 5000
    res["td0"] = {}
    plt.figure(figsize=(7, 4.2))
    for alpha in (0.5, 0.1, 0.01, "1/n"):
        V, err = td0(SWWS, alpha, N_EP, rng, v_swws)
        res["td0"][str(alpha)] = {"V": {s: float(V[s]) for s in NON_ABS},
                                  "erreur_finale": float(err[-1]),
                                  "erreur_moy_500_derniers": float(err[-500:].mean())}
        lisse = np.convolve(err, np.ones(20) / 20, mode="valid")  # moyenne glissante
        plt.plot(np.arange(20, N_EP + 1), lisse, label=f"$\\alpha$ = {alpha}")
    plt.yscale("log")
    plt.xlabel("nombre d'épisodes")
    plt.ylabel("$\\max_s |V(s) - v_{SWWS}(s)|$")
    plt.title("Exercice 3.2 (1) : TD(0) pour SWWS\n(moyenne glissante sur 20 épisodes)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex32_td0.png"), dpi=200)
    plt.close()

    # Q2
    N_EP_Q = 20000
    res["q_learning"] = {}
    plt.figure(figsize=(7, 4.2))
    for alpha in (0.5, 0.1, 0.01):
        Q, glout, bon = q_learning(alpha, 0.1, N_EP_Q, rng, pol_opt)
        # premier episode a partir duquel la politique gloutonne reste la bonne
        reste_bon = int(np.argmax(np.cumprod(bon[::-1])[::-1])) if bon[-1] else None
        res["q_learning"][str(alpha)] = {
            "Q": {f"{s},{a}": float(Q[(s, a)]) for s in NON_ABS for a in ACTIONS},
            "politique": glout,
            "bonne_a_la_fin": bool(bon[-1]),
            "episode_stable": reste_bon,
        }
        # moyenne glissante sur 200 episodes sinon c'est illisible
        glissant = np.convolve(bon.astype(float), np.ones(200) / 200, mode="valid")
        plt.plot(np.arange(200, N_EP_Q + 1), glissant, label=f"$\\alpha$ = {alpha}")
    plt.xlabel("nombre d'épisodes")
    plt.ylabel("part des épisodes où la politique\ngloutonne est optimale")
    plt.title("Exercice 3.2 (2) : Q-learning, $\\varepsilon$ = 0.1")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex32_qlearning.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "ex32.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

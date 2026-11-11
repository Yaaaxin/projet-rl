# -*- coding: utf-8 -*-
"""
Exercice 3.1 : le robot vert (Exemple 2.2 du chapitre 1).

r_w = 1, r_s = 5, r_b = -10, gamma = 0.8, alpha = 0.3, beta = 0.6.
Remarque : dans le poly la panne donne « -r_b ». Avec r_b = -10 ça ferait +10,
ce qui n'a pas de sens (le robot serait content de tomber en panne...). On a
donc pris -10 comme récompense quand il tombe en panne.

1. itération de politique (avec le modèle)
2. sans modèle : (a) Monte-Carlo, (b) TD(0), (c) online MC eps-greedy

Lancer :  python ex31_robot.py
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

# le modele, on s'en sert seulement en Q1 et pour verifier a la fin
GAMMA = 0.8
ALPHA, BETA = 0.3, 0.6
R_S, R_W, R_PANNE = 5.0, 1.0, -10.0

HIGH, LOW = 0, 1
NOMS_ETATS = ["high", "low"]
ACTIONS = {HIGH: ["search", "wait"], LOW: ["search", "wait", "recharge"]}

# TRANSITIONS[(s, a)] = liste de (proba, etat suivant, recompense)
TRANSITIONS = {
    (HIGH, "search"):   [(ALPHA, HIGH, R_S), (1 - ALPHA, LOW, R_S)],
    (HIGH, "wait"):     [(1.0, HIGH, R_W)],
    (LOW, "search"):    [(BETA, LOW, R_S), (1 - BETA, HIGH, R_PANNE)],
    (LOW, "wait"):      [(1.0, LOW, R_W)],
    (LOW, "recharge"):  [(1.0, HIGH, 0.0)],
}

# les 3 politiques de la Q2 (action en high / action en low)
POLITIQUES = {
    "s/w":  {HIGH: "search", LOW: "wait"},
    "s/s":  {HIGH: "search", LOW: "search"},
    "s/re": {HIGH: "search", LOW: "recharge"},
}
GRAINE = 31


def r_barre(s, a):
    """récompense moyenne R1(s, a)"""
    return sum(p * r for p, _, r in TRANSITIONS[(s, a)])


def valeur_exacte(politique):
    """Résout v = R_Pi + gamma P_Pi v (inversion d'un système 2x2)."""
    P = np.zeros((2, 2))
    R = np.zeros(2)
    for s in (HIGH, LOW):
        a = politique[s]
        R[s] = r_barre(s, a)
        for p, s2, _ in TRANSITIONS[(s, a)]:
            P[s, s2] += p
    return np.linalg.solve(np.eye(2) - GAMMA * P, R)


def q_depuis_v(v):
    """q_v(s, a) = R1(s, a) + gamma * E[v(S1)]"""
    return {(s, a): r_barre(s, a) + GAMMA * sum(p * v[s2] for p, s2, _ in TRANSITIONS[(s, a)])
            for s in (HIGH, LOW) for a in ACTIONS[s]}


# Q1 : iteration de politique
def iteration_politique(depart):
    politique = dict(depart)
    historique = []
    while True:
        v = valeur_exacte(politique)  # évaluation
        q = q_depuis_v(v)
        nouvelle = {}
        for s in (HIGH, LOW):  # amélioration gloutonne
            meilleure = max(ACTIONS[s], key=lambda a: q[(s, a)])
            # si egalité on garde l'action actuelle, sinon ça peut boucler a l'infini
            if np.isclose(q[(s, meilleure)], q[(s, politique[s])]):
                meilleure = politique[s]
            nouvelle[s] = meilleure
        historique.append({"politique": f"{politique[HIGH]}/{politique[LOW]}",
                           "v": [round(float(x), 4) for x in v]})
        if nouvelle == politique:  # plus rien ne change donc on s'arrete
            return politique, v, historique
        politique = nouvelle


# simulateur : c'est la seule chose qu'on a le droit d'utiliser en Q2
def un_pas(s, a, rng):
    """Tire l'état suivant et la récompense (comme si on regardait le robot)."""
    u = rng.random()
    cumul = 0.0
    for p, s2, r in TRANSITIONS[(s, a)]:
        cumul += p
        if u < cumul:
            return s2, r
    return s2, r  # au cas ou cumul = 0.9999... a cause des arrondis


# Q2a : Monte-Carlo
T_HORIZON = 60  # 0.8^60 ~ 1.5e-6, la fin de la trajectoire compte presque plus


def retour_mc(politique, s, rng, T=T_HORIZON):
    """G^T(s) = sum_{k<T} gamma^k R_{k+1} en partant de s."""
    G, remise = 0.0, 1.0
    for _ in range(T):
        s, r = un_pas(s, politique[s], rng)
        G += remise * r
        remise *= GAMMA
    return G


def monte_carlo(politique, n_traj, rng):
    """Pour chaque état de départ : moyenne de n_traj retours.
    On garde aussi la moyenne courante pour tracer la convergence."""
    courbes, estimations, demi_ic = {}, {}, {}
    for s in (HIGH, LOW):
        G = np.array([retour_mc(politique, s, rng) for _ in range(n_traj)])
        courbes[s] = np.cumsum(G) / np.arange(1, n_traj + 1)
        estimations[s] = G.mean()
        demi_ic[s] = 1.96 * G.std(ddof=1) / np.sqrt(n_traj)  # interval de confiance a 95%
    return estimations, demi_ic, courbes


# Q2b : TD(0)
def td0(politique, eta, n_episodes, rng, T=50, v_exacte=None):
    """TD(0) : V(S) <- V(S) + eta (R + gamma V(S') - V(S)).

    Le robot n'a pas d'état absorbant, donc on coupe en épisodes de T pas,
    chacun partant d'un état tiré au hasard. C'est necessaire pour s/w :
    en low le robot attend pour toujours, sinon on reverrait jamais high.
    eta peut être un nombre, ou "1/n" (pas 1 / nombre de visites de l'état).
    """
    V = np.zeros(2)
    visites = np.zeros(2)
    erreurs = []
    for _ in range(n_episodes):
        s = rng.integers(2)
        for _ in range(T):
            s2, r = un_pas(s, politique[s], rng)
            visites[s] += 1
            pas = 1.0 / visites[s] if eta == "1/n" else eta
            V[s] += pas * (r + GAMMA * V[s2] - V[s])
            s = s2
        if v_exacte is not None:
            erreurs.append(np.max(np.abs(V - v_exacte)))
    return V, np.array(erreurs)


# Q2c : online MC eps-greedy (algo 4 du chap 3)
def mc_eps_greedy(eps, n_traj, rng, T=100, t_max_visite=60, recompense_immediate=False):
    """Contrôle par Monte-Carlo avec une politique eps-greedy.

    Dans le poly, G = « récompense immédiatement obtenue en quittant (s, a) ».
    On a essayé ça (recompense_immediate=True) mais on trouve s/w : la
    politique ne regarde pas l'avenir. Par défaut on prend donc le vrai
    retour G_t = sum_k gamma^k R_{t+k+1}, calculé sur la fin de la trajectoire.
    On ne garde que les visites t < t_max_visite pour que le retour soit
    calculé sur au moins 40 pas (0.8^40 ~ 1e-4).
    """
    Q = {(s, a): 0.0 for s in (HIGH, LOW) for a in ACTIONS[s]}
    # au lieu de garder toute la liste Returns(s, a) on met juste la moyenne a jour
    # (ça donne la meme chose et c'est beaucoup plus rapide)
    nb_retours = {cle: 0 for cle in Q}
    gloutonne = {HIGH: "search", LOW: "search"}  # politique de depart
    suivi = []  # pour la figure

    def choisir(s):
        # pi(a|s) = 1 - eps + eps/|A(s)| pour l'action gloutonne, eps/|A(s)| sinon
        if rng.random() < eps:
            return ACTIONS[s][rng.integers(len(ACTIONS[s]))]
        return gloutonne[s]

    for _ in range(n_traj):
        # 1) on genere une trajectoire avec la politique actuel
        s = rng.integers(2)
        traj = []
        for _ in range(T):
            a = choisir(s)
            s2, r = un_pas(s, a, rng)
            traj.append((s, a, r))
            s = s2
        # 2) on calcul les retours en partant de la fin
        G = 0.0
        G_t = [0.0] * T
        for t in range(T - 1, -1, -1):
            G = traj[t][2] + GAMMA * G
            G_t[t] = G
        # 3) MAJ de Q = moyenne des retours (toutes les visites)
        for t in range(t_max_visite):
            s_t, a_t, r_t = traj[t]
            cible = r_t if recompense_immediate else G_t[t]
            nb_retours[(s_t, a_t)] += 1
            # moyenne incrementale : nouvelle = ancienne + (cible - ancienne) / n
            Q[(s_t, a_t)] += (cible - Q[(s_t, a_t)]) / nb_retours[(s_t, a_t)]
        # 4) amelioration : on prend l'action gloutonne dans les etats visités
        for s_vu in {x[0] for x in traj[:t_max_visite]}:
            gloutonne[s_vu] = max(ACTIONS[s_vu], key=lambda a: Q[(s_vu, a)])
        suivi.append({cle: Q[cle] for cle in Q})
    return Q, gloutonne, suivi


def main():
    rng = np.random.default_rng(GRAINE)
    res = {}

    # Q1
    pol, v_star, hist = iteration_politique({HIGH: "wait", LOW: "wait"})
    q_star = q_depuis_v(v_star)
    res["iteration_politique"] = hist
    res["v_star"] = [float(x) for x in v_star]
    res["q_star"] = {f"{NOMS_ETATS[s]},{a}": float(v) for (s, a), v in q_star.items()}
    res["valeurs_exactes"] = {nom: [float(x) for x in valeur_exacte(p)]
                              for nom, p in POLITIQUES.items()}

    # Q2a : Monte-Carlo
    N_TRAJ = 2000
    res["monte_carlo"] = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, (nom, p) in zip(axes, POLITIQUES.items()):
        est, ic, courbes = monte_carlo(p, N_TRAJ, rng)
        exact = valeur_exacte(p)
        res["monte_carlo"][nom] = {NOMS_ETATS[s]: [float(est[s]), float(ic[s])] for s in (HIGH, LOW)}
        for s, couleur in ((HIGH, "tab:blue"), (LOW, "tab:orange")):
            ax.plot(np.arange(1, N_TRAJ + 1), courbes[s], color=couleur,
                    label=f"MC, départ {NOMS_ETATS[s]}")
            ax.axhline(exact[s], color=couleur, ls="--", lw=1)
        ax.set_title(f"politique {nom}")
        ax.set_xlabel("nombre de trajectoires")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("estimation de $v_\\Pi(s)$")
    axes[0].legend(fontsize=9)
    fig.suptitle("Exercice 3.1 (a) : Monte-Carlo (pointillés = valeur exacte)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIG, "ex31_mc.png"), dpi=200)
    plt.close(fig)

    # Q2b : TD(0)
    N_EP = 2000
    res["td0"] = {}
    plt.figure(figsize=(7, 4.2))
    for eta in (0.5, 0.1, 0.01, "1/n"):
        res["td0"][str(eta)] = {}
        for nom, p in POLITIQUES.items():
            exact = valeur_exacte(p)
            V, err = td0(p, eta, N_EP, rng, v_exacte=exact)
            res["td0"][str(eta)][nom] = {"V": [float(x) for x in V],
                                         "erreur_finale": float(err[-1]),
                                         "erreur_moy_500_derniers": float(err[-500:].mean())}
            if nom == "s/re":
                # moyenne glissante sur 20 episodes sinon on voit que du bruit
                lisse = np.convolve(err, np.ones(20) / 20, mode="valid")
                plt.plot(np.arange(20, N_EP + 1), lisse, label=f"$\\eta$ = {eta}")
    plt.yscale("log")
    plt.xlabel("nombre d'épisodes (50 pas chacun)")
    plt.ylabel("$\\max_s |V(s) - v_\\Pi(s)|$")
    plt.title("Exercice 3.1 (b) : TD(0) pour s/re, erreur selon le pas $\\eta$\n"
              "(moyenne glissante sur 20 épisodes)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex31_td0.png"), dpi=200)
    plt.close()

    # Q2c : MC eps-greedy, avec le vrai retour puis avec la recompense immediate du poly
    Q, glout, suivi = mc_eps_greedy(0.1, 3000, rng)
    res["mc_eps_greedy"] = {"Q": {f"{NOMS_ETATS[s]},{a}": float(v) for (s, a), v in Q.items()},
                            "politique": f"{glout[HIGH]}/{glout[LOW]}"}
    Qi, glout_i, _ = mc_eps_greedy(0.1, 3000, rng, recompense_immediate=True)
    res["mc_eps_greedy_recompense_immediate"] = {
        "Q": {f"{NOMS_ETATS[s]},{a}": float(v) for (s, a), v in Qi.items()},
        "politique": f"{glout_i[HIGH]}/{glout_i[LOW]}"}

    plt.figure(figsize=(7, 4.2))
    for a, couleur in zip(ACTIONS[LOW], ("tab:red", "tab:green", "tab:blue")):
        plt.plot([x[(LOW, a)] for x in suivi], color=couleur, label=f"Q(low, {a})")
        plt.axhline(q_star[(LOW, a)], color=couleur, ls=":", lw=1)
    plt.xlabel("nombre de trajectoires")
    plt.ylabel("estimation de Q(low, a)")
    plt.title("Exercice 3.1 (c) : MC $\\varepsilon$-greedy ($\\varepsilon$ = 0.1), état low\n"
              "(pointillés : $q^*$ calculé avec le modèle)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "ex31_mc_controle.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "ex31.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Pourquoi le pas 1/n etait si lent dans le projet ? (exercices 3.1 et 3.2)
#
# Dans le rapport on a vu que TD(0) avec le pas 1/n converge en theorie mais en pratique
# il est tres lent. Ici on teste d'autres pas :
#   - constant 0.01 (le meilleur du projet, pour comparer)
#   - 1/n                      (celui du projet)
#   - 1/n^w  avec w = 0.8, 0.6 (pas "polynomial", Even-Dar & Mansour 2003)
#   - c/(n + c - 1) avec c = 5 = 1/(1 - gamma)
# ou n = nombre de visites de l'etat (ou du couple (s, a) pour Q-learning).
#
# On reprend le modele directement dans les fichiers du projet (dossier code/).
# Pour que ça aille vite, toutes les regles de pas et toutes les graines tournent
# en meme temps dans des tableaux numpy.
#
# Lancer :  python pas_apprentissage.py   (1-2 min)

import json
import os
import sys

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

GAMMA = 0.8
NB_GRAINES = 30
GRAINE = 33

# regle de pas : alpha = a / (n + b)^w
REGLES = {
    "constant 0.01": (0.01, 0, 0.0),
    "1/n": (1.0, 0, 1.0),
    "1/n^0.8": (1.0, 0, 0.8),
    "1/n^0.6": (1.0, 0, 0.6),
    "5/(n+4)": (5.0, 4, 1.0),
}
COULEURS = {"constant 0.01": "tab:green", "1/n": "tab:red", "1/n^0.8": "tab:blue",
            "1/n^0.6": "tab:purple", "5/(n+4)": "tab:orange"}


def parametres_runs():
    """chaque run = (une regle, une graine). On renvoie a, b, w de longueur nb_regles * NB_GRAINES."""
    a, b, w = [], [], []
    for (aa, bb, ww) in REGLES.values():
        a += [aa] * NB_GRAINES
        b += [bb] * NB_GRAINES
        w += [ww] * NB_GRAINES
    return np.array(a), np.array(b, dtype=float), np.array(w)


def tables(transitions, nb_etats):
    """transitions[s] = liste de (p, s2, r) pour l'action choisie en s -> tableaux (2 issues max)."""
    P1 = np.ones(nb_etats)  # proba de la 1ere issue
    S2 = np.zeros((nb_etats, 2), dtype=int)
    RW = np.zeros((nb_etats, 2))
    for s, issues in transitions.items():
        assert len(issues) <= 2
        P1[s] = issues[0][0]
        for k, (p, s2, r) in enumerate(issues):
            S2[s, k], RW[s, k] = s2, r
        if len(issues) == 1:
            S2[s, 1], RW[s, 1] = S2[s, 0], RW[s, 0]
    return P1, S2, RW


def grille_log(T, nb=150):
    return np.unique(np.logspace(1, np.log10(T), nb).astype(int))


# partie 0 : le calcul a la main
def biais_seul():
    """Un seul etat qui boucle sur lui meme, sans bruit : e_n = e_{n-1} (1 - (1-gamma) alpha_n).
    Ça montre juste la vitesse a laquelle on oublie l'initialisation V = 0."""
    n = np.arange(1, 10**6 + 1)
    res = {}
    for nom, (a, b, w) in REGLES.items():
        alpha = np.minimum(1.0, a / (n + b) ** w)
        res[nom] = np.cumprod(1 - (1 - GAMMA) * alpha)
    return n, res


# partie 1 : TD(0) robot, politique s/re
def td_robot(T=250_000, long_episode=50):
    rng = np.random.default_rng(GRAINE)
    pol = robot.POLITIQUES["s/re"]
    v = robot.valeur_exacte(pol)
    P1, S2, RW = tables({s: robot.TRANSITIONS[(s, pol[s])] for s in (0, 1)}, 2)
    a, b, w = parametres_runs()
    nr = len(a)
    lignes = np.arange(nr)
    V = np.zeros((nr, 2))
    n = np.zeros((nr, 2))
    grille = grille_log(T)
    erreurs = []
    j = 0
    for t in range(1, T + 1):
        if (t - 1) % long_episode == 0:
            s = rng.integers(2, size=nr)  # nouvel episode, depart au hasard comme dans le projet
        k = (rng.random(nr) >= P1[s]).astype(int)
        s2, rec = S2[s, k], RW[s, k]
        n[lignes, s] += 1
        alpha = a / (n[lignes, s] + b) ** w
        V[lignes, s] += alpha * (rec + GAMMA * V[lignes, s2] - V[lignes, s])
        s = s2
        if j < len(grille) and t == grille[j]:
            erreurs.append(np.abs(V - v).max(axis=1))
            j += 1
    return grille, np.array(erreurs)  # erreurs[temps, run]


# partie 2 : TD(0) etudiant, politique SWWS
def td_etudiant(T=200_000):
    rng = np.random.default_rng(GRAINE + 1)
    v = etu.valeur_exacte(etu.SWWS)
    trans = {}
    for s in etu.NON_ABS:
        trans[s] = [(p, s2, etu.r[s2]) for s2, p in etu.P[(s, etu.SWWS[s])].items()]
    P1, S2, RW = tables(trans, 8)
    v_vec = np.zeros(8)
    for s in etu.NON_ABS:
        v_vec[s] = v[s]
    absorbant = np.zeros(8, dtype=bool)
    absorbant[list(etu.ABSORBANTS)] = True
    a, b, w = parametres_runs()
    nr = len(a)
    lignes = np.arange(nr)
    V = np.zeros((nr, 8))
    n = np.zeros((nr, 8))
    s = rng.integers(1, 5, size=nr)
    grille = grille_log(T)
    erreurs = []
    j = 0
    for t in range(1, T + 1):
        k = (rng.random(nr) >= P1[s]).astype(int)
        s2, rec = S2[s, k], RW[s, k]
        n[lignes, s] += 1
        alpha = a / (n[lignes, s] + b) ** w
        V[lignes, s] += alpha * (rec + GAMMA * V[lignes, s2] - V[lignes, s])
        # quand on tombe dans 5, 6 ou 7 l'episode est fini : on repart au hasard dans {1,2,3,4}
        s = np.where(absorbant[s2], rng.integers(1, 5, size=nr), s2)
        if j < len(grille) and t == grille[j]:
            erreurs.append(np.abs(V[:, 1:5] - v_vec[1:5]).max(axis=1))
            j += 1
    return grille, np.array(erreurs)


# partie 3 : Q-learning etudiant
def q_learning_etudiant(T=300_000, eps=0.1):
    rng = np.random.default_rng(GRAINE + 2)
    pol_opt, v_opt, _ = etu.iteration_politique({s: "W" for s in etu.NON_ABS})
    v_vec = np.array([0.0] + [v_opt[s] for s in etu.NON_ABS])  # indices 0..4, on ignore 0
    opt = np.array([0] + [etu.ACTIONS.index(pol_opt[s]) for s in etu.NON_ABS])
    # transitions pour les deux actions
    P1 = np.ones((8, 2))
    S2 = np.zeros((8, 2, 2), dtype=int)
    RW = np.zeros((8, 2, 2))
    for s in etu.NON_ABS:
        for ia, act in enumerate(etu.ACTIONS):
            issues = list(etu.P[(s, act)].items())
            P1[s, ia] = issues[0][1]
            for k, (s2, p) in enumerate(issues):
                S2[s, ia, k], RW[s, ia, k] = s2, etu.r[s2]
            if len(issues) == 1:
                S2[s, ia, 1], RW[s, ia, 1] = S2[s, ia, 0], RW[s, ia, 0]
    absorbant = np.zeros(8, dtype=bool)
    absorbant[list(etu.ABSORBANTS)] = True

    a, b, w = parametres_runs()
    nr = len(a)
    lignes = np.arange(nr)
    Q = np.zeros((nr, 8, 2))
    n = np.zeros((nr, 8, 2))
    s = rng.integers(1, 5, size=nr)
    grille = grille_log(T)
    err_v, bonne = [], []
    j = 0
    for t in range(1, T + 1):
        qs = Q[lignes, s]
        glouton = np.argmax(qs + 1e-9 * rng.random(qs.shape), axis=1)
        act = np.where(rng.random(nr) < eps, rng.integers(2, size=nr), glouton)
        k = (rng.random(nr) >= P1[s, act]).astype(int)
        s2, rec = S2[s, act, k], RW[s, act, k]
        n[lignes, s, act] += 1
        alpha = a / (n[lignes, s, act] + b) ** w
        cible = rec + GAMMA * Q[lignes, s2].max(axis=1)  # Q reste a 0 sur les absorbants
        Q[lignes, s, act] += alpha * (cible - Q[lignes, s, act])
        s = np.where(absorbant[s2], rng.integers(1, 5, size=nr), s2)
        if j < len(grille) and t == grille[j]:
            err_v.append(np.abs(Q[:, 1:5].max(axis=2) - v_vec[1:5]).max(axis=1))
            bonne.append((Q[:, 1:5].argmax(axis=2) == opt[1:5]).all(axis=1))
            j += 1
    return grille, np.array(err_v), np.array(bonne)


def par_regle(x):
    """x[temps, run] -> dict regle -> x[temps, graine]"""
    return {nom: x[:, i * NB_GRAINES:(i + 1) * NB_GRAINES] for i, nom in enumerate(REGLES)}


def trace(ax, grille, x, titre, ylabel, log=True):
    for nom, xx in par_regle(x).items():
        m = xx.mean(axis=1)
        bas, haut = np.percentile(xx, [10, 90], axis=1)
        ax.plot(grille, m, color=COULEURS[nom], label=nom)
        ax.fill_between(grille, bas, haut, color=COULEURS[nom], alpha=0.15)
    ax.set_xscale("log")
    if log:
        ax.set_yscale("log")
    ax.set_title(titre, fontsize=10)
    ax.set_xlabel("nombre de mises a jour")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")


def resume(grille, x, seuil):
    out = {}
    for nom, xx in par_regle(x).items():
        fin = xx[-1]
        m = xx.mean(axis=1)
        sous = np.flatnonzero(m < seuil)
        out[nom] = {"erreur_finale_moy": float(fin.mean()), "erreur_finale_sd": float(fin.std(ddof=1)),
                    f"premier_t_erreur_moy_sous_{seuil}": int(grille[sous[0]]) if len(sous) else None}
    return out


def main():
    res = {"nb_graines": NB_GRAINES}

    # partie 0
    n, biais = biais_seul()
    plt.figure(figsize=(7, 4.2))
    for nom, e in biais.items():
        plt.plot(n[::50], e[::50], color=COULEURS[nom], label=nom)
    plt.plot(n[::50], n[::50] ** -0.2 * biais["1/n"][-1] / (1e6 ** -0.2), "k:", lw=1, label="pente n^(-0.2)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-8, 1.5)
    plt.xlabel("n (nombre de visites)")
    plt.ylabel("e_n / e_0")
    plt.title("Sans bruit : vitesse a laquelle on oublie V = 0 (gamma = 0.8)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "pas_biais_seul.png"), dpi=200)
    plt.close()
    res["biais_seul_n_1e5"] = {nom: float(e[10**5 - 1]) for nom, e in biais.items()}

    # parties 1, 2, 3
    g1, e1 = td_robot()
    print("robot ok")
    g2, e2 = td_etudiant()
    print("etudiant TD ok")
    g3, e3, bon3 = q_learning_etudiant()
    print("Q-learning ok")
    res["td_robot_s_re"] = resume(g1, e1, 0.5)
    res["td_etudiant_SWWS"] = resume(g2, e2, 2.0)
    res["q_learning_etudiant"] = resume(g3, e3, 2.0)
    for nom, bb in par_regle(bon3.astype(float)).items():
        res["q_learning_etudiant"][nom]["part_graines_politique_optimale_fin"] = float(bb[-1].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    trace(axes[0], g1, e1, "TD(0), robot, politique s/re", "max_s |V(s) - v(s)|")
    trace(axes[1], g2, e2, "TD(0), etudiant, politique SWWS", "max_s |V(s) - v(s)|")
    trace(axes[2], g3, e3, "Q-learning, etudiant (eps = 0.1)", "max_s |max_a Q(s,a) - v*(s)|")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Erreur selon la regle de pas (moyenne sur {NB_GRAINES} graines, bande = 10%-90%)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOSSIER_FIG, "pas_erreurs.png"), dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 4.2))
    for nom, bb in par_regle(bon3.astype(float)).items():
        plt.plot(g3, bb.mean(axis=1), color=COULEURS[nom], label=nom)
    plt.xscale("log")
    plt.xlabel("nombre de mises a jour")
    plt.ylabel("part des graines ou la politique gloutonne\nest W/W/W/R")
    plt.title("Q-learning (etudiant) : quand est-ce qu'on trouve la bonne politique ?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER_FIG, "pas_qlearning_politique.png"), dpi=200)
    plt.close()

    with open(os.path.join(DOSSIER_RES, "pas_apprentissage.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

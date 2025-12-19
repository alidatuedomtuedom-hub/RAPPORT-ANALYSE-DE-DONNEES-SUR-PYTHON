#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import os
OUTDIR = "img_finale"
os.makedirs(OUTDIR, exist_ok=True)

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print("\n Bienvenue à la séance 4 de Karlaaaa") 
print(dist_names)

def moyenne(data):
    """Calcule la moyenne d'un échantillon (liste ou array)."""
    data = np.asarray(data, dtype=float)
    return float(np.sum(data) / len(data))

def ecart_type(data):
    """Calcule l'écart-type (population) d'un échantillon : sqrt( (1/n) * Σ(x - m)^2 )."""
    data = np.asarray(data, dtype=float)
    m = moyenne(data)
    var = np.sum((data - m) ** 2) / len(data)
    return float(np.sqrt(var))

# 1) Distributions de variables DISCRÈTES : Dirac, Uniforme discrète, Binomiale, Poisson, Zipf-Mandelbrot

n_disc = 1000

# Dirac
dirac_val = 5
ech_dirac = np.full(n_disc, dirac_val)

# Uniforme discrète (dé 1..6)
ech_uniforme_disc = scipy.stats.randint.rvs(1, 7, size=n_disc)

# Binomiale
ech_binomiale = scipy.stats.binom.rvs(n=15, p=0.6, size=n_disc)

# Poisson (discrète)
mu_poisson = 5
ech_poisson_disc = scipy.stats.poisson.rvs(mu=mu_poisson, size=n_disc)

# Zipf–Mandelbrot : p(k) ∝ (k + q)^(-s), k = 1..Kmax
# (PMF construite manuellement car pas de loi SciPy dédiée systématiquement disponible)
s = 2.0
q = 1.0
Kmax = 100
k = np.arange(1, Kmax + 1)
pmf = 1 / (k + q) ** s
pmf = pmf / pmf.sum()
ech_zipf_mandelbrot = np.random.choice(k, size=n_disc, p=pmf)

discretes = {
    "Dirac": ech_dirac,
    "Uniforme_discrete": ech_uniforme_disc,
    "Binomiale": ech_binomiale,
    "Poisson": ech_poisson_disc,
    "Zipf_Mandelbrot": ech_zipf_mandelbrot
}

print(" Contrôles rapides (discrètes)")
for nom, data in discretes.items():
    print(f"{nom:18s} : uniques={len(np.unique(data))}, min={data.min()}, max={data.max()}")


# 2) Distributions de variables CONTINUES: Poisson (demandé à intégrer), Normale, Log-normale, Uniforme, Chi2, Pareto
n_cont = 2000

ech_poisson_cont = scipy.stats.poisson.rvs(mu=mu_poisson, size=n_cont)

ech_normale = scipy.stats.norm.rvs(loc=0, scale=1, size=n_cont)
ech_lognormale = scipy.stats.lognorm.rvs(s=0.5, loc=0, scale=1, size=n_cont)
ech_uniforme = scipy.stats.uniform.rvs(loc=0, scale=1, size=n_cont)
ech_chi2 = scipy.stats.chi2.rvs(df=3, size=n_cont)
ech_pareto = scipy.stats.pareto.rvs(b=3, loc=0, scale=1, size=n_cont)

continues = {
    "Poisson_continue": ech_poisson_cont,
    "Normale": ech_normale,
    "Lognormale": ech_lognormale,
    "Uniforme": ech_uniforme,
    "Chi2": ech_chi2,
    "Pareto": ech_pareto
}

print("Contrôles rapides (continues)")
for nom, data in continues.items():
    print(f"{nom:18s} : mean={np.mean(data):.3f}, std={np.std(data):.3f}")

# 3) Visualisations

# A) Discrètes : barres des effectifs (ou fréquences)
for nom, data in discretes.items():
    valeurs, effectifs = np.unique(data, return_counts=True)

    plt.figure()
    plt.bar(valeurs, effectifs)

    plt.title(f"Loi {nom} (discrète)")
    plt.xlabel("Valeurs")
    plt.ylabel("Effectifs")

    if nom == "Zipf_Mandelbrot":
        plt.yscale("log")
        plt.ylabel("Effectifs (échelle log)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"discrete_{nom}.png"), dpi=200)
    plt.close()

# B) Continues : histogramme + densité théorique (sauf Poisson_continue)
for nom, data in continues.items():
    plt.figure()

    if nom == "Poisson_continue":
        # Fréquences empiriques
        valeurs, effectifs = np.unique(data, return_counts=True)
        freqs = effectifs / effectifs.sum()
        plt.bar(valeurs, freqs, alpha=0.55, label="Fréquences (échantillon)")

        # Courbe de PMF théorique tracée "en ligne"
        kk = np.arange(0, int(np.max(valeurs)) + 1)
        pmf_theo = scipy.stats.poisson.pmf(kk, mu=mu_poisson)
        plt.plot(kk, pmf_theo, linewidth=2, label="PMF théorique (tracée)")

        plt.title("Poisson (visualisation demandée en section 'continue')")
        plt.xlabel("k")
        plt.ylabel("Probabilité / fréquence")
        plt.legend()

    else:
        # Histogramme empirique
        plt.hist(data, bins=40, density=True, alpha=0.6, label="Histogramme (échantillon)")

        x = np.linspace(np.min(data), np.max(data), 500)

        if nom == "Normale":
            pdf = scipy.stats.norm.pdf(x, loc=0, scale=1)
        elif nom == "Lognormale":
            pdf = scipy.stats.lognorm.pdf(x, s=0.5, loc=0, scale=1)
        elif nom == "Uniforme":
            pdf = scipy.stats.uniform.pdf(x, loc=0, scale=1)
        elif nom == "Chi2":
            pdf = scipy.stats.chi2.pdf(x, df=3)
        elif nom == "Pareto":
            pdf = scipy.stats.pareto.pdf(x, b=3, loc=0, scale=1)
        else:
            pdf = None

        if pdf is not None:
            plt.plot(x, pdf, linewidth=2, label="Densité théorique (pdf)")

        plt.title(f"Loi {nom} (continue)")
        plt.xlabel("x")
        plt.ylabel("Densité")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"continue_{nom}.png"), dpi=200)
    plt.close()

print(f" Figures enregistrées dans le dossier : {OUTDIR}")

print(" Moyenne et écart-type des distributions")
for nom, data in {**discretes, **continues}.items():
    m = moyenne(data)
    s = ecart_type(data)
    print(f"{nom:18s} -> moyenne = {m:.3f} | écart-type = {s:.3f}")
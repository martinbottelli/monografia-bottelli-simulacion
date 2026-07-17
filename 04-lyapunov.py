"""
04_lyapunov.py
--------------
Calcula el mayor exponente de Lyapunov del reservorio de prediccion mediante
el metodo de Benettin (evolucion de un vector tangente con el jacobiano del
lazo cerrado).

Si la reconstruccion del clima es correcta, este exponente deberia aproximar
el mayor exponente de Lyapunov del sistema de Lorenz (~0.906), tal como
argumentan los autores del paper.

Se calcula para varios valores de la fuerza de entrada, para observar la
relacion entre la calidad de la reconstruccion y el exponente obtenido.

Uso:
    python 04_lyapunov.py
"""

import numpy as np
import matplotlib.pyplot as plt

import comun


SEEDS = [1, 2, 3]
N_TRAIN = 15000


def main():
    datos, media, desvio = comun.preparar_datos(t_max=600.0)

    print("Calculando el mayor exponente de Lyapunov del reservorio...")
    sigmas = [0.02, 0.05, 0.1, 0.2, 0.4]
    medias, desvios = [], []
    for s in sigmas:
        vals = []
        for seed in SEEDS:
            M, Win, Wout, r_last = comun.entrenar_reservorio_completo(
                datos, N_TRAIN, sigma_in=s, seed=seed)
            le = comun.mayor_exponente_lyapunov(M, Win, Wout, r_last)
            vals.append(le)
        medias.append(np.mean(vals)); desvios.append(np.std(vals))
        print(f"  sigma_in={s:.2f}: exponente = {np.mean(vals):.3f} "
              f"+/- {np.std(vals):.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigmas, medias, yerr=desvios, marker="o", capsize=4,
                color="tab:red", label="Reservorio")
    ax.axhline(comun.LYAP_LORENZ, color="tab:blue", ls="--",
               label=f"Lorenz real ({comun.LYAP_LORENZ})")
    ax.set_xscale("log")
    ax.set_xlabel("Fuerza de entrada (sigma_in)")
    ax.set_ylabel("Mayor exponente de Lyapunov")
    ax.set_title("Exponente de Lyapunov del reservorio de prediccion")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("figuras/lyapunov.png", dpi=150)
    print("Figura guardada en figuras/lyapunov.png")


if __name__ == "__main__":
    main()
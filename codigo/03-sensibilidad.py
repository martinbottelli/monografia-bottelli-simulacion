"""
03-sensibilidad.py
------------------
Analisis de sensibilidad del reservorio. Estudia como afectan a la calidad de
la prediccion dos parametros clave:

  1. La fuerza de entrada (sigma_in), analoga al parametro sigma del paper.
  2. La regularizacion (beta) de la regresion ridge.

Para cada valor se mide el tiempo de prediccion valido, promediado sobre
varios reservorios aleatorios distintos.

Uso:
    python codigo/03-sensibilidad.py

Nota: este script entrena muchos reservorios, puede tardar un par de minutos.
"""

import numpy as np
import matplotlib.pyplot as plt

import comun


SEEDS = [1, 2, 3]       # reservorios aleatorios para promediar
N_TRAIN = 15000
N_PRED = 1000


def medir(datos, sigma_in, beta):
    """Tiempo de prediccion valido promedio para (sigma_in, beta)."""
    real = datos[N_TRAIN:N_TRAIN + N_PRED]
    tiempos = []
    for seed in SEEDS:
        M, Win, Wout, r_last = comun.entrenar_reservorio_completo(
            datos, N_TRAIN, sigma_in=sigma_in, beta=beta, seed=seed)
        pred = comun.predecir(M, Win, Wout, r_last, N_PRED)
        tiempos.append(comun.tiempo_valido(pred, real) * comun.DT)
    return np.mean(tiempos), np.std(tiempos)


def main():
    datos, media, desvio = comun.preparar_datos(t_max=600.0)

    # =================================================================
    # Barrido de la fuerza de entrada
    # =================================================================
    print("Barrido de la fuerza de entrada (sigma_in)...")
    sigmas = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    medias_s, desvios_s = [], []
    for s in sigmas:
        m, d = medir(datos, sigma_in=s, beta=comun.BETA_DEF)
        medias_s.append(m); desvios_s.append(d)
        print(f"  sigma_in={s:.2f}: tiempo valido = {m:.1f} +/- {d:.1f} u.t.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigmas, medias_s, yerr=desvios_s, marker="o", capsize=4,
                color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("Fuerza de entrada (sigma_in)")
    ax.set_ylabel("Tiempo de prediccion valido [u.t.]")
    ax.set_title("Sensibilidad a la fuerza de entrada")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    ruta = comun.ruta_figura("sensibilidad_sigma.png")
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada en {ruta}")

    # =================================================================
    # Efecto de la regularizacion
    # =================================================================
    print("\nEfecto de la regularizacion (beta)...")
    betas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
    medias_b, desvios_b = [], []
    for b in betas:
        m, d = medir(datos, sigma_in=comun.SIGMA_IN_DEF, beta=b)
        medias_b.append(m); desvios_b.append(d)
        print(f"  beta={b:.0e}: tiempo valido = {m:.1f} +/- {d:.1f} u.t.")

    # En el eje x usamos un valor pequeno para representar beta=0 en escala log
    betas_plot = [1e-9 if b == 0 else b for b in betas]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(betas_plot, medias_b, yerr=desvios_b, marker="s", capsize=4,
                color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("Regularizacion (beta)  [beta=0 representado en 1e-9]")
    ax.set_ylabel("Tiempo de prediccion valido [u.t.]")
    ax.set_title("Efecto de la regularizacion")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    ruta = comun.ruta_figura("sensibilidad_beta.png")
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada en {ruta}")


if __name__ == "__main__":
    main()
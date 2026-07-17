"""
02_reservoir.py
---------------
Nucleo del trabajo. Construye el reservorio, lo entrena sobre la serie de
Lorenz y lo hace predecir en lazo cerrado. Genera las tres figuras centrales:

  1. Prediccion de corto plazo: senal real vs predicha (reproduce la Fig. 5).
  2. Reconstruccion del atractor: mariposa real vs reconstruida.
  3. Mapa de retorno de Poincare de los maximos de z (reproduce la Fig. 6).

Uso:
    python 02_reservoir.py
"""

import numpy as np
import matplotlib.pyplot as plt

import comun


def main():
    # --- Datos ---
    datos, media, desvio = comun.preparar_datos(t_max=600.0)
    n_train = 15000
    n_pred = 6000
    real = datos[n_train:n_train + n_pred]

    # --- Entrenamiento del reservorio ---
    print("Construyendo y entrenando el reservorio...")
    M, Win, Wout, r_last = comun.entrenar_reservorio_completo(
        datos, n_train, seed=1)

    # --- Prediccion en lazo cerrado ---
    print("Prediciendo en lazo cerrado...")
    pred = comun.predecir(M, Win, Wout, r_last, n_pred)

    # Desnormalizar para graficar en las unidades originales
    real_d = real * desvio + media
    pred_d = pred * desvio + media

    tv = comun.tiempo_valido(pred, real)
    print(f"Tiempo de prediccion valido: {tv} pasos "
          f"({tv * comun.DT:.1f} unidades de tiempo)")

    # =================================================================
    # Figura 1: prediccion de corto plazo de z(t)
    # =================================================================
    t = np.arange(n_pred) * comun.DT
    n_zoom = int(15.0 / comun.DT)  # primeras 15 unidades de tiempo
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[:n_zoom], real_d[:n_zoom, 2], label="Real", color="tab:blue")
    ax.plot(t[:n_zoom], pred_d[:n_zoom, 2], label="Prediccion",
            color="tab:red", ls="--")
    ax.axvline(tv * comun.DT, color="gray", ls=":", label="Fin prediccion valida")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("z(t)")
    ax.set_title("Prediccion de corto plazo")
    ax.legend()
    ax.grid(True, ls=":")
    fig.tight_layout()
    fig.savefig("figuras/prediccion_corto_plazo.png", dpi=150)
    print("Figura guardada en figuras/prediccion_corto_plazo.png")

    # =================================================================
    # Figura 2: atractor real vs reconstruido
    # =================================================================
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(real_d[:, 0], real_d[:, 1], real_d[:, 2], lw=0.4, color="tab:blue")
    ax1.set_title("Atractor real")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(pred_d[:, 0], pred_d[:, 1], pred_d[:, 2], lw=0.4, color="tab:red")
    ax2.set_title("Atractor reconstruido por el reservorio")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    fig.tight_layout()
    fig.savefig("figuras/atractor_reconstruido.png", dpi=150)
    print("Figura guardada en figuras/atractor_reconstruido.png")

    # =================================================================
    # Figura 3: mapa de retorno de Poincare de los maximos de z
    # =================================================================
    zmax_real = comun.maximos_locales(real_d[:, 2])
    zmax_pred = comun.maximos_locales(pred_d[:, 2])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(zmax_real[:-1], zmax_real[1:], s=8, color="tab:blue",
               label="Real")
    ax.scatter(zmax_pred[:-1], zmax_pred[1:], s=8, color="tab:red",
               label="Prediccion", alpha=0.6)
    ax.set_xlabel(r"$z_n^{max}$")
    ax.set_ylabel(r"$z_{n+1}^{max}$")
    ax.set_title("Mapa de retorno de Poincare")
    ax.legend()
    ax.grid(True, ls=":")
    fig.tight_layout()
    fig.savefig("figuras/mapa_poincare.png", dpi=150)
    print("Figura guardada en figuras/mapa_poincare.png")


if __name__ == "__main__":
    main()
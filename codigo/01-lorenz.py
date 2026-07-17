"""
01-lorenz.py
------------
Genera la serie temporal del sistema de Lorenz y grafica el atractor
(la "mariposa"). Tambien muestra como dos trayectorias con condiciones
iniciales casi identicas se separan con el tiempo (efecto mariposa).

Corresponde a la primera parte del capitulo de Implementacion de la monografia.

Uso:
    python codigo/01-lorenz.py
"""

import numpy as np
import matplotlib.pyplot as plt

import comun


def graficar_atractor(tray, nombre="lorenz_atractor.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tray[:, 0], tray[:, 1], tray[:, 2], lw=0.5, color="tab:blue")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Atractor de Lorenz")
    fig.tight_layout()
    ruta = comun.ruta_figura(nombre)
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada en {ruta}")


def graficar_divergencia(nombre="lorenz_divergencia.png"):
    tray1 = comun.generar_lorenz(t_max=50.0, estado_inicial=(1.0, 1.0, 1.0),
                                 descartar=0)
    tray2 = comun.generar_lorenz(t_max=50.0, estado_inicial=(1.0, 1.0, 1.0 + 1e-8),
                                 descartar=0)
    m = min(len(tray1), len(tray2))
    dist = np.sqrt(((tray1[:m] - tray2[:m]) ** 2).sum(axis=1))
    t = np.arange(m) * comun.DT

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(t, dist, color="tab:red")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Distancia entre trayectorias (escala log)")
    ax.set_title("Sensibilidad a las condiciones iniciales (efecto mariposa)")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    ruta = comun.ruta_figura(nombre)
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada en {ruta}")


def main():
    tray = comun.generar_lorenz(t_max=100.0)
    graficar_atractor(tray)
    graficar_divergencia()


if __name__ == "__main__":
    main()
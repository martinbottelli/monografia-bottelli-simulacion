"""
comun.py
--------
Funciones compartidas por todos los scripts del trabajo.

Contiene:
  - La resolucion de la ruta de la carpeta de figuras.
  - La integracion del sistema de Lorenz.
  - La preparacion de los datos (normalizacion y particion).
  - La construccion, escucha (listening) y entrenamiento del reservorio.
  - La prediccion en lazo cerrado (predicting).
  - El calculo del mayor exponente de Lyapunov del reservorio de prediccion.

Se implementa la version en tiempo discreto del reservorio (Ecuacion 2 del
paper), que segun los propios autores es la adecuada para reservorios de
software. Los datos se normalizan (media 0, desvio 1) para que el metodo sea
numericamente robusto.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


# =====================================================================
#  Rutas
# =====================================================================
# La ruta se resuelve a partir de la ubicacion de este archivo y no del
# directorio de trabajo, para que los scripts se puedan ejecutar desde
# cualquier lado.
DIR_FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def ruta_figura(nombre):
    """Ruta absoluta de una figura, creando la carpeta si hace falta."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    return DIR_FIGURAS / nombre


# =====================================================================
#  Parametros por defecto (validados para que el reservorio reconstruya
#  el atractor de Lorenz de forma estable)
# =====================================================================
DT = 0.02          # paso de muestreo (unidades de tiempo de Lorenz)
N_DEF = 600        # dimension del reservorio (cantidad de neuronas)
RHO_DEF = 0.9      # radio espectral de la matriz interna (igual que el paper)
DENS_DEF = 0.02    # densidad de conexiones (igual que el paper)
ALPHA_DEF = 1.0    # tasa de fuga (1.0 = reservorio sin fuga)
SIGMA_IN_DEF = 0.1 # fuerza de entrada (input strength)
BETA_DEF = 1e-6    # regularizacion de la regresion ridge
WASHOUT_DEF = 300  # pasos iniciales del reservorio que se descartan

# Mayor exponente de Lyapunov del sistema de Lorenz (valor de referencia)
LYAP_LORENZ = 0.906


# =====================================================================
#  Sistema de Lorenz
# =====================================================================
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz(t, estado):
    """Derivadas del sistema de Lorenz."""
    x, y, z = estado
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def generar_lorenz(t_max=600.0, dt=DT, estado_inicial=(1.0, 1.0, 1.0),
                   descartar=500):
    """
    Integra el sistema de Lorenz con Runge-Kutta y devuelve la serie
    temporal como un arreglo (T, 3). Descarta un transitorio inicial para
    quedarse sobre el atractor.
    """
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz, (0, t_max), list(estado_inicial), t_eval=t_eval,
                    method="RK45", rtol=1e-10, atol=1e-10)
    return sol.y.T[descartar:]


def preparar_datos(t_max=600.0):
    """
    Genera la serie de Lorenz y la normaliza. Devuelve:
      - datos normalizados (T, 3)
      - media y desvio (para poder desnormalizar despues)
    """
    U = generar_lorenz(t_max=t_max)
    media, desvio = U.mean(axis=0), U.std(axis=0)
    U_norm = (U - media) / desvio
    return U_norm, media, desvio


# =====================================================================
#  Reservorio
# =====================================================================
def construir_reservorio(N=N_DEF, rho=RHO_DEF, densidad=DENS_DEF,
                         sigma_in=SIGMA_IN_DEF, seed=1):
    """
    Construye las matrices del reservorio, aleatorias y fijas:
      - M:   matriz interna, rala, reescalada al radio espectral 'rho'.
      - Win: matriz de pesos de entrada.
    """
    rng = np.random.default_rng(seed)
    M = rng.uniform(-1, 1, (N, N)) * (rng.random((N, N)) < densidad)
    radio = np.max(np.abs(np.linalg.eigvals(M)))
    M *= rho / radio
    Win = rng.uniform(-sigma_in, sigma_in, (N, 3))
    return M, Win


def _features(R):
    """Aplica la no linealidad al estado: eleva al cuadrado las coordenadas
    de indice par. Esto rompe la simetria del sistema de Lorenz y mejora
    mucho la reconstruccion (truco estandar en reservoir computing)."""
    Q = R.copy()
    Q[:, ::2] = Q[:, ::2] ** 2
    return Q


def _feature_vec(r):
    """Version de _features para un unico vector de estado."""
    q = r.copy()
    q[::2] = q[::2] ** 2
    return q


def escuchar(M, Win, entradas, alpha=ALPHA_DEF):
    """
    Etapa de escucha (listening): maneja el reservorio con la serie real y
    devuelve todos los estados y el ultimo estado.
    """
    N = M.shape[0]
    r = np.zeros(N)
    R = np.zeros((len(entradas), N))
    for n in range(len(entradas)):
        r = (1 - alpha) * r + alpha * np.tanh(M @ r + Win @ entradas[n])
        R[n] = r
    return R, r


def entrenar(R, objetivo, beta=BETA_DEF, washout=WASHOUT_DEF):
    """
    Etapa de entrenamiento (training): ajusta la matriz de salida Wout por
    regresion ridge para que, a partir del estado del reservorio, prediga el
    siguiente valor de la serie.
    """
    Q = _features(R[washout:-1])
    Y = objetivo[washout + 1:]
    F = Q.shape[1]
    Wout = np.linalg.solve(Q.T @ Q + beta * np.eye(F), Q.T @ Y).T
    return Wout


def predecir(M, Win, Wout, r_inicial, n_pasos, alpha=ALPHA_DEF):
    """
    Etapa de prediccion (predicting): el reservorio corre en lazo cerrado,
    usando su propia salida como entrada. Devuelve la serie predicha (n_pasos, 3).
    """
    r = r_inicial.copy()
    preds = np.zeros((n_pasos, 3))
    for n in range(n_pasos):
        u = Wout @ _feature_vec(r)
        preds[n] = u
        r = (1 - alpha) * r + alpha * np.tanh(M @ r + Win @ u)
    return preds


def entrenar_reservorio_completo(datos, n_train, N=N_DEF, rho=RHO_DEF,
                                 sigma_in=SIGMA_IN_DEF, beta=BETA_DEF,
                                 alpha=ALPHA_DEF, seed=1):
    """
    Atajo que hace todo el pipeline: construye, escucha y entrena.
    Devuelve M, Win, Wout y el ultimo estado del reservorio.
    """
    train = datos[:n_train]
    M, Win = construir_reservorio(N=N, rho=rho, sigma_in=sigma_in, seed=seed)
    R, r_last = escuchar(M, Win, train, alpha=alpha)
    Wout = entrenar(R, train, beta=beta)
    return M, Win, Wout, r_last


# =====================================================================
#  Exponente de Lyapunov del reservorio de prediccion (metodo de Benettin)
# =====================================================================
def mayor_exponente_lyapunov(M, Win, Wout, r_inicial, alpha=ALPHA_DEF,
                             n_pasos=6000, n_trans=500, dt=DT):
    """
    Estima el mayor exponente de Lyapunov del reservorio de prediccion
    evolucionando un vector tangente con el jacobiano del lazo cerrado y
    promediando su tasa de crecimiento logaritmica.

    Si el clima se reconstruye correctamente, este valor deberia aproximar
    el mayor exponente de Lyapunov del sistema de Lorenz (~0.906).
    """
    N = M.shape[0]
    r = r_inicial.copy()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)

    acumulado, cuenta = 0.0, 0
    for n in range(n_pasos):
        # Derivada de las features respecto del estado:
        #   d_i = 1        para indice impar
        #   d_i = 2 r_i    para indice par
        d = np.ones(N)
        d[::2] = 2 * r[::2]

        u = Wout @ _feature_vec(r)
        z = M @ r + Win @ u
        sech2 = 1 - np.tanh(z) ** 2

        # Jacobiano aplicado al vector tangente (sin formar la matriz NxN):
        #   J v = (1-a) v + a * sech2 * (M v + Win Wout (d*v))
        Jv = (1 - alpha) * v + alpha * sech2 * (M @ v + Win @ (Wout @ (d * v)))

        # Avanzar el estado del reservorio
        r = (1 - alpha) * r + alpha * np.tanh(z)

        # Renormalizar el vector tangente y acumular
        norma = np.linalg.norm(Jv)
        v = Jv / norma
        if n >= n_trans:
            acumulado += np.log(norma)
            cuenta += 1

    # Dividir por dt para pasar al tiempo continuo de Lorenz
    return acumulado / cuenta / dt


# =====================================================================
#  Utilidades
# =====================================================================
def maximos_locales(serie):
    """Devuelve los maximos locales de una serie 1D (para el mapa de Poincare)."""
    idx = np.where((serie[1:-1] > serie[:-2]) & (serie[1:-1] > serie[2:]))[0] + 1
    return serie[idx]


def tiempo_valido(pred, real, umbral=0.4):
    """
    Cantidad de pasos durante los cuales la prediccion sigue a la trayectoria
    real antes de superar un umbral de error normalizado.
    """
    m = min(len(pred), len(real))
    escala = np.sqrt((real ** 2).sum(axis=1)).mean()
    err = np.sqrt(((pred[:m] - real[:m]) ** 2).sum(axis=1)) / escala
    malos = np.where(err > umbral)[0]
    return malos[0] if len(malos) > 0 else m
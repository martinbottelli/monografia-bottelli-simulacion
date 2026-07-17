# Attractor Reconstruction by Machine Learning — Reproducción

Reproducción en Python de los experimentos numéricos del paper:

> Z. Lu, B. R. Hunt, E. Ott. *Attractor reconstruction by machine learning*.
> Chaos: An Interdisciplinary Journal of Nonlinear Science, 28(6), 061104 (2018).

Trabajo final de la materia **Simulación (75.26)** — FIUBA.

El trabajo aplica *reservoir computing* (una técnica de aprendizaje automático)
para reconstruir el atractor del sistema de Lorenz a partir de una serie
temporal, sin conocer las ecuaciones del sistema.

## Requisitos

- Python 3.10 o superior
- Las dependencias listadas en `requirements.txt`

## Instalación

```bash
git clone <URL-del-repo>
cd reservoir-lorenz
pip install -r requirements.txt
```

## Cómo replicar los resultados

Todo el código está en la carpeta `codigo/`. El código común (sistema de Lorenz,
construcción y entrenamiento del reservorio, cálculo de exponentes de Lyapunov)
está en `codigo/comun.py`. Cada script numerado usa ese módulo para generar las
figuras de una sección del informe. Las figuras se guardan en la carpeta
`figuras/`, que se crea sola si no existe. Se ejecutan en orden:

```bash
python codigo/01-lorenz.py
python codigo/02-reservoir.py
python codigo/03-sensibilidad.py
python codigo/04-lyapunov.py
```

Los scripts resuelven la carpeta de figuras a partir de su propia ubicación, así
que se pueden ejecutar desde cualquier directorio.

| Script | Qué genera | Figuras |
|--------|------------|---------|
| `01-lorenz.py` | El atractor de Lorenz y la separación de dos trayectorias próximas (efecto mariposa). | `lorenz_atractor.png`, `lorenz_divergencia.png` |
| `02-reservoir.py` | Predicción de corto plazo, reconstrucción del atractor y mapa de retorno de Poincaré. | `prediccion_corto_plazo.png`, `atractor_reconstruido.png`, `mapa_poincare.png` |
| `03-sensibilidad.py` | Efecto de la fuerza de entrada y de la regularización sobre la calidad de la predicción. | `sensibilidad_sigma.png`, `sensibilidad_beta.png` |
| `04-lyapunov.py` | Mayor exponente de Lyapunov del reservorio de predicción, comparado con el de Lorenz. | `lyapunov.png` |

Los scripts `03` y `04` entrenan varios reservorios y pueden tardar un par de
minutos.

## Parámetros principales

Los parámetros del reservorio están centralizados en `codigo/comun.py` (dimensión
`N=600`, radio espectral `0.9`, densidad `0.02`, fuerza de entrada `0.1` y
regularización `1e-6`). Modificándolos ahí se puede repetir cualquier
experimento con otra configuración.

## Estructura del repositorio

```
reservoir-lorenz/
├── codigo/
│   ├── comun.py             # Funciones compartidas (Lorenz + reservorio + Lyapunov)
│   ├── 01-lorenz.py         # Sistema de Lorenz y efecto mariposa
│   ├── 02-reservoir.py      # Predicción, reconstrucción del atractor y Poincaré
│   ├── 03-sensibilidad.py   # Sensibilidad a la fuerza de entrada y la regularización
│   └── 04-lyapunov.py       # Exponente de Lyapunov del reservorio
├── figuras/                 # Figuras generadas por los scripts
├── requirements.txt         # Dependencias
└── README.md
```
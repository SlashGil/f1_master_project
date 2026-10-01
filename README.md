# 🏎️ Formula 1 Race Win Prediction: A Machine Learning Benchmark (1950–2024)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-5%2F5%20Passing-brightgreen?logo=checkmarx&logoColor=white)](tests/test_temporal_integrity.py)
[![Data Leakage](https://img.shields.io/badge/Data%20Leakage-0%25%20(Audited)-success)](INFORME_REFACTORIZACION_Y_AUDITORIA_METODOLOGICA.md)
[![Observations](https://img.shields.io/badge/Canonical%20Dataset-25%2C121%20Races-red)](data/preparar_datos_tesis.py)
[![GitHub Pages](https://img.shields.io/badge/Live%20Showcase-GitHub%20Pages-00f0ff?logo=github)](docs/index.html)

> **Trabajo de Tesis de Maestría:** *Predicción de Victorias en Fórmula 1 mediante Aprendizaje Automático: Un Estudio Comparativo de Modelos Predictivos.*  
> **Autor:** Salvador Romero Gil  
> **Área:** Ciencia de Datos, Aprendizaje Automático y Series de Tiempo  
> **Periodo Histórico:** 74 Temporadas Oficiales de la FIA (1950 – 2024)  

---

## 🌐 Aplicación Web Interactiva & Showcase en GitHub Pages

Este repositorio incluye una **aplicación web interactiva completa** lista para ser desplegada en **GitHub Pages**. Permite explorar los resultados empíricos, inspeccionar las curvas ROC y matrices de confusión, y utilizar un **Simulador Pit Wall en Vivo** para calcular probabilidades de victoria en tiempo real:

👉 **[Ver Aplicación Web / Dashboard Interactivo (`docs/index.html`)](docs/index.html)**

### ¿Cómo activar GitHub Pages en tu repositorio?
1. Dirígete a tu repositorio en GitHub y haz clic en **Settings** (Configuración).
2. En la barra lateral izquierda, selecciona **Pages**.
3. En **Build and deployment > Source**, selecciona **Deploy from a branch**.
4. En **Branch**, selecciona `main` y en la carpeta elige **/docs**. Haz clic en **Save**.
5. ¡Listo! Tu sitio estará activo en: `https://<tu-usuario>.github.io/<nombre-del-repo>/`.

---

## 📌 1. Descripción Ejecutiva del Proyecto

El objetivo científico de este proyecto consiste en evaluar la capacidad predictiva de algoritmos de aprendizaje automático para pronosticar al **ganador absoluto** ($P1$) de un Gran Premio de Fórmula 1 utilizando **únicamente información disponible antes de la largada** (clasificación, características del constructor, edad del piloto y contexto de la temporada).

### Los Dos Grandes Desafíos Metodológicos
1. **Desbalance Extremo de Clases (1:21.3):**  
   En cada Gran Premio compiten entre 20 y 24 pilotos y solo uno resulta vencedor. La clase positiva (`win = 1`) representa únicamente el **4.49%** del total histórico (1,128 victorias frente a 23,993 no victorias), lo que provoca el colapso de clasificadores estándar si no se compensa adecuadamente la función de pérdida o el umbral de decisión.
2. **Causalidad Temporal Estricta y Riesgo de *Data Leakage*:**  
   A diferencia de los problemas tabulares comunes, los eventos de Fórmula 1 son estrictamente secuenciales. El uso inadvertido de particiones aleatorias (`train_test_split(shuffle=True)`), identificadores correlacionados cronológicamente (`raceId`) o métricas intra-carrera invalida científicamente los resultados al filtrar información del futuro hacia el pasado.

---

## 🛡️ 2. Arquitectura de Datos y Protocolo Anti-Leakage

Tras una rigurosa auditoría metodológica interna, el pipeline de datos fue refactorizado con las siguientes garantías:

```
                                      FUENTES HISTÓRICAS (ERGAST API)
                      [results.csv, races.csv, drivers.csv, constructors.csv]
                                                 │
                                                 ▼
                                     data/preparar_datos_tesis.py
                                                 │
                                 ┌───────────────┴───────────────┐
                                 ▼                               ▼
                      VARIABLES PRE-CARRERA             VARIABLES PROHIBIDAS
                      • grid (Parrilla)                 ❌ fastestLapSpeed / LapTime
                      • age (Edad del piloto)           ❌ milliseconds_pit_stop
                      • constructorId (Escudería)       ❌ positionOrder / Points
                      • round, year, nationality        ❌ raceId (Excluido de X)
                                 │
                                 ▼
                     CANONICAL DATASET (25,121 Observaciones)
                                 │
                 ┌───────────────┴───────────────────────────────┐
                 ▼                                               ▼
         ENTRENAMIENTO (80%)                              PRUEBA CIEGA (20%)
         20,096 Observaciones                             5,025 Observaciones
     1950-05-13 a 2012-11-04 (Abu Dabi)               2012-11-04 a 2024-11-03 (São Paulo)
```

### Invariantes de Integridad Garantizadas
* **Frontera Temporal Causal:**  
  $$\max(\text{fecha}_{\text{train}}) \le \min(\text{fecha}_{\text{test}})$$
  Ninguna carrera del conjunto de prueba antecede a las carreras de entrenamiento.
* **Validación Cruzada Temporal (`TimeSeriesSplit`):**  
  La optimización de hiperparámetros se realizó con 5 pliegues temporales expansivos donde para cada pliegue $k$:
  $$\max(\text{fecha}_{\text{train}}^{(k)}) \le \min(\text{fecha}_{\text{val}}^{(k)})$$
* **Exclusión de `raceId`:**  
  `raceId` es un identificador sintético autoincremental que actúa como proxy artificial del tiempo. Se excluyó de la matriz de predictores $X$ en todos los modelos para evitar pseudocorrelaciones espurias.

---

## 🤖 3. Modelos Predictivos en el Alcance Oficial

El estudio evalúa de forma homogénea cuatro paradigmas de modelado sobre el dataset canónico de 25,121 filas:

| Script | Paradigma Algorítmico | Propósito y Configuración |
|---|---|---|
| [`modelos/1_regresion_lineal.py`](modelos/1_regresion_lineal.py) | **Regresión Lineal MCO** | Modelo base para cuantificar la magnitud y polaridad de los coeficientes estandarizados pre-carrera (sin `raceId`). |
| [`modelos/2_random_forest.py`](modelos/2_random_forest.py) | **Random Forest (Ensemble)** | Árboles de decisión con balanceo de clases y búsqueda por malla vía `TimeSeriesSplit`. Regularización `min_samples_leaf=2`. |
| [`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py) | **Red Neuronal Densa (MLP)** | Red profunda PyTorch (128-64-32) con corrección de pérdida ponderada `nn.BCELoss(reduction='none')` y respaldo en Scikit-Learn. |
| [`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py) | **Regresión Logística** | Modelo probabilístico para inferencia causal y cálculo de *Odds Ratios* ($\text{OR}$) e intervalos de confianza. |

---

## 📊 4. Cuadro Comparativo de Resultados Oficiales

Todas las métricas fueron obtenidas evaluando los modelos de manera ciega sobre el conjunto de prueba independiente ($N_{\text{test}} = 5,025$ observaciones correspondientes a las temporadas 2012 a 2024):

| Modelo Predictivo | Muestra Train / Test | F1-Score | AUC-ROC | AUC-PR | Recall | Precision | Conclusión y Estado Metodológico |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🏆 **Random Forest (GridSearch)** | 20,096 / 5,025 | **0.3349** | **0.8869** | **0.2748** | **83.94%** | **20.98%** | **Modelo Campeón:** Máximo equilibrio entre capacidad discriminativa global y recuperación de victorias. Hiperparámetros óptimos: `n_estimators: 100`, `max_depth: 12`, `min_samples_leaf: 2`. |
| **Random Forest (Baseline)** | 20,096 / 5,025 | 0.3139 | 0.8855 | 0.2608 | 86.35% | 19.15% | Alta sensibilidad ante la clase victoria; detecta a la gran mayoría de ganadores a costa de falsos positivos controlables. |
| **Regresión Logística** | 20,096 / 5,025 | 0.2640 | 0.8220 | 0.1792 | 55.82% | 17.29% | Altamente explicable: la posición de largada (`grid`) reduce el Odds Ratio en un factor de $0.0797$ por posición ($\beta = -2.5294$). |
| **Regresión Lineal** | 20,096 / 5,025 | 0.0000 | 0.8121 | 0.2327 | 0.00% | 0.00% | Su buen AUC (0.8121) demuestra capacidad ordinal, pero el umbral rígido $0.5$ colapsa ante el desbalance del 4.49%. |
| **Perceptrón Multicapa (MLP)** | 20,096 / 5,025 | 0.0000 | 0.7135 | 0.1074 | 0.00% | 0.00% | Capacidad discriminativa global verificada ($AUC = 0.7135$); requiere calibración fina del umbral de corte. |

---

## 🧪 5. Suite de Pruebas Unitarias de Integridad

El repositorio incorpora una suite de pruebas automatizadas en [`tests/test_temporal_integrity.py`](tests/test_temporal_integrity.py) para certificar de forma demostrable la validez del pipeline ante comités académicos:

```bash
# Ejecutar todas las pruebas unitarias
python -m unittest discover tests -v
```

### Pruebas Implementadas:
1. `test_01_chronological_split_boundary`: Certifica matemáticamente que $\max(\text{train}) \le \min(\text{test})$ en la partición 80/20.
2. `test_02_time_series_split_expanding_folds`: Simula y valida los 5 pliegues expansivos de `TimeSeriesSplit`, garantizando que ninguna carrera de validación anteceda al entrenamiento.
3. `test_03_no_prohibited_in_race_features`: Inspecciona el dataset para descartar la presencia de variables intra-carrera o post-carrera.
4. `test_04_raceid_excluded_from_model_features`: Aplica análisis estático sobre el código de los cuatro modelos oficiales para constatar la exclusión programática de `raceId`.
5. `test_05_unified_dataset_size`: Verifica la integridad muestral exacta con $N = 25,121$ observaciones.

**Resultado:**
```text
Ran 5 tests in 0.653s
OK
[OK] Frontera Temporal 80/20 Validada (Train: 20,096 | Test: 5,025)
[OK] Validando 5 folds de TimeSeriesSplit
[OK] Cero variables post-carrera prohibidas en el dataset
[OK] Exclusión de 'raceId' confirmada en los 4 modelos oficiales
[OK] Integridad Muestral: 25,121 observaciones confirmadas
```

---

## 📁 6. Estructura del Repositorio

A continuación se detalla el contenido y la función de cada archivo y directorio en el proyecto:

```
f1_master_project/
│
├── .githooks/                               # Hooks de control de versiones y gobernanza
│   └── pre-push                             # Hook para restringir pushes directos a main solo a usuarios autorizados
│
├── data/                                    # Pipeline y fuentes de datos
│   ├── preparar_datos_tesis.py              # Script oficial de extracción y generación del dataset pre-carrera
│   └── dataset_tesis_f1.csv                 # Dataset unificado oficial (25,121 filas) [Regenerable]
│
├── docs/                                    # Sitio web interactivo para GitHub Pages
│   ├── index.html                           # Aplicación web completa con Dashboard y Simulador Pit Wall
│   └── assets/                              # Gráficos de alta resolución exportados de los modelos
│       ├── roc_rf_grid.png                  # Curva ROC de Random Forest Optimizado
│       ├── confusion_matrix_rf_grid.png     # Matriz de Confusión de Random Forest
│       ├── feature_importance_rf.png        # Importancia de variables (Gini Impurity)
│       ├── coeficientes_logistica.png       # Coeficientes estandarizados de Regresión Logística
│       ├── odds_ratios_logistica.png        # Gráfico de Odds Ratios multiplicativos
│       ├── roc_logistica.png                # Curva ROC de Regresión Logística
│       ├── roc_lineal.png                   # Curva ROC de Regresión Lineal
│       ├── roc_mlp.png                      # Curva ROC de Red Neuronal MLP
│       └── confusion_matrix_mlp.png         # Matriz de Confusión de Red Neuronal MLP
│
├── modelos/                                 # Scripts oficiales de entrenamiento y evaluación
│   ├── 1_regresion_lineal.py                # Modelo 1: Regresión Lineal pre-carrera
│   ├── 2_random_forest.py                   # Modelo 2: Random Forest (Baseline + GridSearch)
│   ├── 3_perceptron_multicapa.py            # Modelo 3: Perceptrón Multicapa (PyTorch / Scikit-Learn)
│   └── 4_regresion_logistica.py             # Modelo 4: Regresión Logística con Odds Ratios
│
├── resultados_*/                            # Artefactos, métricas y gráficos generados por cada modelo
│   ├── resultados_regresion_lineal/         # CSVs de métricas, coeficientes y curvas ROC
│   ├── resultados_random_forest_baseline/   # Métricas y matrices de confusión del baseline
│   ├── resultados_random_forest_grid_search/# Parámetros óptimos, curvas PR y ROC del campeón
│   ├── resultados_regresion_logistica/      # Tablas de coeficientes y Odds Ratios
│   └── resultados_mlp/                      # Curvas de pérdida por época y métricas del MLP
│
├── tests/                                   # Pruebas automatizadas de aseguramiento de calidad
│   └── test_temporal_integrity.py           # Suite unitaria que certifica la no-fuga de datos
│
├── legacy/                                  # Scripts preliminares archivados por obsolescencia metodológica
│   ├── 1_preparacion_dataset.py             # Versión preliminar antigua
│   ├── modelo_regresion_lineal.py           # Versión previa con split aleatorio
│   ├── f1_win.py                            # Script exploratorio inicial
│   └── README.md                            # Justificación de archivo y estado metodológico
│
├── index.html                               # Redirección universal a docs/index.html para GitHub Pages
├── INFORME_REFACTORIZACION_Y_AUDITORIA_METODOLOGICA.md  # Monografía técnica y académica formal
├── DOCUMENTACION_TRANSFORMACION_DATOS.md    # Especificación detallada de variables y transformaciones
├── .gitattributes                           # Configuración de normalización de saltos de línea (LF)
├── .gitignore                               # Exclusiones de Git (entornos, cachés y artefactos fuera de scope)
└── README.md                                # Este documento
```

---

## 🚀 7. Guía de Instalación y Ejecución

### 7.1. Clonar el Repositorio
```bash
git clone https://github.com/<tu-usuario>/f1_master_project.git
cd f1_master_project
```

### 7.2. Crear y Activar Entorno Virtual
```bash
# En Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# En Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 7.3. Instalar Dependencias
```bash
pip install -r - << 'EOF'
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
imbalanced-learn>=0.12.0
joblib>=1.3.0
EOF
```

### 7.4. Ejecutar la Suite de Pruebas Unitarias
```bash
python -m unittest discover tests -v
```

### 7.5. Entrenar y Evaluar los Modelos
```bash
# Cambiar al directorio de modelos
cd modelos

# Ejecutar los modelos oficiales
python 1_regresion_lineal.py
python 2_random_forest.py
python 3_perceptron_multicapa.py
python 4_regresion_logistica.py
```
Los resultados (gráficos en PNG y métricas en CSV) se depositarán automáticamente en sus respectivos directorios `resultados_*`.

---

## 🔒 8. Gobernanza en Git: Protección de la Rama `main`

El repositorio cuenta con una política de protección mediante el hook de pre-inserción [`.githooks/pre-push`](.githooks/pre-push):
* **Comportamiento:** Intercepta cualquier intento de ejecutar `git push origin main` y valida la identidad del usuario contra una lista de control de acceso (*whitelist*).
* **Configuración del Usuario Autorizado:**
  ```bash
  # Registrar usuario con permiso de push directo a main
  git config --local hooks.allowed-main-users "tu_usuario_github"
  ```
  Cualquier usuario no autorizado que intente hacer un push directo a `main` verá la operación abortada automáticamente, debiendo colaborar mediante ramas secundarias y Pull Requests.

---

## 📄 9. Citación y Atribución Académica

Si utilizas este repositorio, código o metodología en investigaciones académicas, por favor cita este trabajo como:

```bibtex
@mastersthesis{romero2026f1prediction,
  author       = { Salvador Romero Gil },
  title        = {Predicci{\'o}n de Victorias en F{\'o}rmula 1 mediante Aprendizaje Autom{\'a}tico: Un Estudio Comparativo de Modelos Predictivos},
  school       = { Universidad Autonoma del Estado de Puebla } ,
  year         = { 2026 },
  month        = {Septiembre},
  note         = {Repositorio de C{\'o}digo Abierto con Suite de Pruebas Unitarias e Integridad Temporal}
}
```

---

<p align="center">
  <sub>Desarrollado con rigor científico y pasión por el automovilismo &bull; 1950–2024 &bull; F1 Master Project</sub>
</p>

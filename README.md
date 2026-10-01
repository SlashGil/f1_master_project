# 🏎️ Formula 1 Race Win Prediction: A Machine Learning Benchmark (1950–2024)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-5%2F5%20Passing-brightgreen?logo=checkmarx&logoColor=white)](tests/test_temporal_integrity.py)
[![Data Leakage](https://img.shields.io/badge/Data%20Leakage-0%25%20(Audited)-success)](INFORME_REFACTORIZACION_Y_AUDITORIA_METODOLOGICA.md)
[![Canonical Dataset](https://img.shields.io/badge/Canonical%20Dataset-25%2C121%20Races-red)](data/preparar_datos_tesis.py)
[![GitHub Pages](https://img.shields.io/badge/Live%20Showcase-GitHub%20Pages-00f0ff?logo=github)](docs/index.html)

> **Trabajo de Tesis de Maestría:** *Predicción de Victorias en Fórmula 1 mediante Aprendizaje Automático: Un Estudio Comparativo de Modelos Predictivos.*  
> **Autor:** Salvador Romero Gil  
> **Área:** Ciencia de Datos, Aprendizaje Automático y Series de Tiempo  
> **Periodo Histórico:** 74 Temporadas Oficiales de la FIA (1950 – 2024)  

---

## 🌐 Aplicación Web Interactiva & Showcase en GitHub Pages

Este repositorio incluye una **aplicación web interactiva completa** diseñada para ser desplegada en **GitHub Pages**. Permite explorar los resultados empíricos, inspeccionar las curvas ROC y matrices de confusión, y utilizar un **Simulador Pit Wall en Vivo** para calcular probabilidades de victoria en tiempo real:

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

Tras una rigurosa auditoría metodológica interna y un estudio de ablación dimensional, el pipeline de datos fue refactorizado con las siguientes garantías:

```
                                      FUENTES HISTÓRICAS (ERGAST API)
                      [results.csv, races.csv, drivers.csv, constructors.csv]
                                                 │
                                                 ▼
                                     data/preparar_datos_tesis.py
                                                 │
                                 ┌───────────────┴───────────────┐
                                 ▼                               ▼
                      VARIABLES PRE-CARRERA             VARIABLES EXCLUIDAS / PROHIBIDAS
                      • grid (Parrilla)                 ❌ fastestLapSpeed / LapTime
                      • age (Edad del piloto)           ❌ milliseconds_pit_stop
                      • round (Ronda de temporada)      ❌ positionOrder / Points
                      • constructorId_* (Escudería)     ❌ raceId (Excluido de X)
                                                        ❌ year (Excluido de X; solo para orden temporal)
                                                        ❌ nationality_* (Excluidas; ruido dimensional)
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
* **Exclusión de `raceId`, `year` y `nationality`:**  
  Se comprobó empíricamente que eliminar `year` (evita extrapolaciones macrotemporales espurias) y `nationality` (elimina 43 columnas binarias esparsas) optimiza la generalización en todos los modelos, reduciendo el espacio de características a **201 variables limpias**.

---

## 🤖 3. Modelos Predictivos en el Alcance Oficial de la Tesis

El estudio evalúa de forma homogénea tres paradigmas de modelado sobre el dataset canónico de 25,121 filas:

| Script | Paradigma Algorítmico | Propósito y Configuración |
|---|---|---|
| [`modelos/2_random_forest.py`](modelos/2_random_forest.py) | **Random Forest (Ensemble)** | Árboles de decisión con balanceo de clases y búsqueda por malla vía `TimeSeriesSplit`. Regularización `min_samples_leaf=2`. |
| [`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py) | **Red Neuronal Densa (MLP)** | Red profunda PyTorch (201-64-32-1) con corrección de pérdida ponderada `nn.BCELoss(reduction='none')` ($w=11.43$). |
| [`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py) | **Regresión Logística** | Modelo probabilístico para inferencia causal, cálculo de *Odds Ratios* ($\text{OR}$) e interpretabilidad física. |

---

## 📊 4. Cuadro Comparativo: Resultados Nuevos vs Resultados Anteriores

Todas las métricas fueron obtenidas evaluando los modelos de manera ciega sobre el conjunto de prueba independiente ($N_{\text{test}} = 5,025$ observaciones correspondientes a las temporadas 2012 a 2024).

### 4.1. Estudio Comparativo: Impacto de la Eliminación de `year` y `nationality`

| Modelo Predictivo | Configuración de Features | F1-Score | AUC-ROC | AUC-PR | Recall | Precision | Delta de Rendimiento ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🏆 **Random Forest (GridSearch)** | Con Year & Nat (245)<br>**Sin Year & Nat (201)** | 0.3349<br>**0.3580** | 0.8869<br>**0.9265** | 0.2748<br>**0.3766** | 83.94%<br>**88.35%** | 20.98%<br>**22.45%** | **Mejora Superior:** $\Delta \text{AUC-ROC} = +0.0396$, $\Delta \text{AUC-PR} = +0.1018$, $\Delta F_1 = +0.0231$. Rescata 220 de 249 victorias reales. |
| **Random Forest (Baseline)** | Con Year & Nat (245)<br>**Sin Year & Nat (201)** | 0.3139<br>**0.3402** | 0.8855<br>**0.9220** | 0.2608<br>**0.3283** | 86.35%<br>**89.16%** | 19.15%<br>**21.02%** | **Mejora General:** $\Delta \text{AUC-ROC} = +0.0365$, $\Delta \text{AUC-PR} = +0.0675$, $\Delta F_1 = +0.0263$. Gran equilibrio y estabilidad. |
| **Regresión Logística** | Con Year & Nat (245)<br>**Sin Year & Nat (201)** | 0.2640<br>**0.3027** | 0.8220<br>**0.9233** | 0.1792<br>**0.3207** | 55.82%<br>**93.98%** | 17.29%<br>**18.04%** | **Salto Notable:** $\Delta \text{AUC-ROC} = +0.1013$, $\Delta \text{Recall} = +38.16\%$. El modelo lineal generalizado pasó a rescatar 234 victorias. |
| **Perceptrón Multicapa (MLP)** | Con Year & Nat (245)<br>**Sin Year & Nat (201)** | 0.1253<br>**0.2069** | 0.6488<br>**0.8818** | 0.0822<br>**0.3531** | 96.39%<br>**97.59%** | 6.70%<br>**11.57%** | **Convergencia Óptima:** $\Delta \text{AUC-ROC} = +0.2330$, $\Delta \text{AUC-PR} = +0.2709$. Los gradientes de PyTorch convergieron de forma balanceada. |

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
4. `test_04_raceid_excluded_from_model_features`: Aplica análisis estático sobre el código de los tres modelos oficiales para constatar la exclusión programática de `raceId`, `year` y `nationality`.
5. `test_05_unified_dataset_size`: Verifica la integridad muestral exacta con $N = 25,121$ observaciones.

**Resultado:**
```text
Ran 5 tests in 0.921s
OK
[OK] Frontera Temporal 80/20 Validada (Train: 20,096 | Test: 5,025)
[OK] Validando 5 folds de TimeSeriesSplit
[OK] Cero variables post-carrera prohibidas en el dataset
[OK] Exclusión de 'raceId', 'year' y 'nationality' confirmada en los 3 modelos oficiales
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
│       ├── roc_mlp.png                      # Curva ROC de Red Neuronal MLP
│       └── confusion_matrix_mlp.png         # Matriz de Confusión de Red Neuronal MLP
│
├── modelos/                                 # Scripts oficiales de entrenamiento y evaluación de la tesis
│   ├── 2_random_forest.py                   # Modelo: Random Forest (Baseline + GridSearch)
│   ├── 3_perceptron_multicapa.py            # Modelo: Perceptrón Multicapa (PyTorch nativo)
│   └── 4_regresion_logistica.py             # Modelo: Regresión Logística con Odds Ratios
│
├── resultados_*/                            # Artefactos, métricas y gráficos generados por cada modelo
│   ├── resultados_random_forest_baseline/   # Métricas y matrices de confusión del baseline
│   ├── resultados_random_forest_grid_search/# Parámetros óptimos, curvas PR y ROC del campeón
│   ├── resultados_regresion_logistica/      # Tablas de coeficientes y Odds Ratios
│   └── resultados_mlp/                      # Pesos .pt, curvas de pérdida y métricas del MLP
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
├── requirements.txt                         # Dependencias del proyecto (PyTorch, Scikit-Learn, Pandas, etc.)
├── index.html                               # Redirección universal a docs/index.html para GitHub Pages
├── INFORME_REFACTORIZACION_Y_AUDITORIA_METODOLOGICA.md  # Monografía técnica y académica formal
├── DOCUMENTACION_TRANSFORMACION_DATOS.md    # Especificación detallada de variables y transformaciones
├── .gitattributes                           # Configuración de normalización de saltos de línea (LF)
├── .gitignore                               # Exclusiones de Git (entornos, modelos no incluidos y dataset grande)
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
pip install -r requirements.txt
```

### 7.4. Ejecutar la Suite de Pruebas Unitarias
```bash
python -m unittest discover tests -v
```

### 7.5. Entrenar y Evaluar los Modelos
```bash
# Cambiar al directorio de modelos
cd modelos

# Ejecutar los modelos oficiales de la tesis
python 2_random_forest.py
python 3_perceptron_multicapa.py
python 4_regresion_logistica.py
```
Los resultados (gráficos en PNG, pesos `.pt` y métricas en CSV) se depositarán automáticamente en sus respectivos directorios `resultados_*`.

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
  author       = {Salvador Romero Gil},
  title        = {Predicci{\'o}n de Victorias en F{\'o}rmula 1 mediante Aprendizaje Autom{\'a}tico: Un Estudio Comparativo de Modelos Predictivos},
  school       = {Programa de Maestr{\'i}a},
  year         = {2026},
  month        = {Septiembre},
  note         = {Repositorio de C{\'o}digo Abierto con Suite de Pruebas Unitarias e Integridad Temporal}
}
```

---

<p align="center">
  <sub>Desarrollado con rigor científico y pasión por el automovilismo &bull; 1950–2024 &bull; F1 Master Project</sub>
</p>

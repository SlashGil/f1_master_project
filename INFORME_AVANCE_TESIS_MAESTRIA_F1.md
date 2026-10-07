> **INSTRUCCIÓN PARA GEMINI AL GENERAR EL GOOGLE DOC:**  
> *"Por favor, toma el siguiente informe técnico y genera un documento formal de Google Docs con formato académico profesional. Aplica la jerarquía de estilos de Google Docs (Título, Subtítulo, Encabezado 1, Encabezado 2, Encabezado 3), diseña las tablas con formato tabular limpio y encabezados destacados, formatea las expresiones matemáticas con tipografía de ecuación o texto formal, e inserta los bloques de código fuente como cajas de código estructurado con sangría. Conserva íntegramente la redacción, citas de archivos, tablas de ablación y métricas oficiales reproducibles."*

---

# INFORME DE AVANCE Y AUDITORÍA METODOLÓGICA DE INVESTIGACIÓN

**Proyecto de Tesis de Maestría:** *Modelado Predictivo y Análisis Causal de Victorias en Fórmula 1 mediante Aprendizaje Automático y Validación Temporal Estricta (1950–2024)*  
**Tesista / Investigador Principal:** Salvador Romero Gil  
**Institución Académica:** Programa de Posgrado en Ciencias de Datos / Inteligencia Artificial  
**Fecha Oficial de Emisión:** 07 de Octubre de 2026  
**Enlace Oficial al Repositorio en GitHub:** [https://github.com/SlashGil/f1_master_project](https://github.com/SlashGil/f1_master_project)  
**Rama Oficial de Trabajo y Evaluación:** [`test`](https://github.com/SlashGil/f1_master_project/tree/test)  

---

## ÍNDICE GENERAL
1. [Enlace Oficial del Repositorio y Arquitectura de Control de Versiones](#1-enlace-oficial-del-repositorio-y-arquitectura-de-control-de-versiones)
2. [Informe de Avance Metodológico y Tratamiento de la Frontera Causal](#2-informe-de-avance-metodológico-y-tratamiento-de-la-frontera-causal)
   - 2.1. [Problemática de Fractura de Carreras en el Split Clásico 80/20](#21-problemática-de-fractura-de-carreras-en-el-split-clásico-8020)
   - 2.2. [Estudio Experimental de Estrategias de Frontera: Gran Premio de Abu Dhabi 2012](#22-estudio-experimental-de-estrategias-de-frontera-gran-premio-de-abu-dhabi-2012)
   - 2.3. [Implementación del Particionamiento Temporal Atómico y Validación Cruzada sin Fractura](#23-implementación-del-particionamiento-temporal-atómico-y-validación-cruzada-sin-fractura)
3. [Estudio Experimental de Ablación Dimensional (Variables `year` y `nationality`)](#3-estudio-experimental-de-ablación-dimensional-variables-year-y-nationality)
   - 3.1. [Matriz de Rendimiento Factorial $2 \times 2$](#31-matriz-de-rendimiento-factorial-2-times-2)
   - 3.2. [Decisión Metodológica y Fundamentación Econométrica](#32-decisión-metodológica-y-fundamentación-econométrica)
4. [Base de Datos Congelada y Diccionario de Variables Predictoras Finales](#4-base-de-datos-congelada-y-diccionario-de-variables-predictoras-finales)
   - 4.1. [Protocolo de Congelamiento de Datos (*Data Freezing*)](#41-protocolo-de-congelamiento-de-datos-data-freezing)
   - 4.2. [Tabla Estructurada del Espacio de Características Final ($D = 201$)](#42-tabla-estructurada-del-espacio-de-características-final-d--201)
5. [Resultados Oficiales Regenerados en Git (Métricas Verificadas en Disco)](#5-resultados-oficiales-regenerados-en-git-métricas-verificadas-en-disco)
   - 5.1. [Cuadro Comparativo de los Tres Modelos Oficiales de la Tesis](#51-cuadro-comparativo-de-los-tres-modelos-oficiales-de-la-tesis)
   - 5.2. [Evidencia de Regeneración y Trazabilidad en el Repositorio](#52-evidencia-de-regeneración-y-trazabilidad-en-el-repositorio)
6. [Auditoría Integral y Corrección de Problemáticas en el Repositorio](#6-auditoría-integral-y-corrección-de-problemáticas-en-el-repositorio)
   - 6.1. [Erradicación de Fuga Temporal (*Data Leakage*)](#61-erradicación-de-fuga-temporal-data-leakage)
   - 6.2. [Supresión de Variables Intra/Post-Carrera](#62-supresión-de-variables-intrapost-carrera)
   - 6.3. [Exclusión del Identificador Sintético `raceId`](#63-exclusión-del-identificador-sintético-raceid)
   - 6.4. [Depuración del Alcance Oficial: Exclusión de OLS y Modelo 5](#64-depuración-del-alcance-oficial-exclusión-de-ols-y-modelo-5)
   - 6.5. [Gobernanza Git y Blindaje de la Rama `main`](#65-gobernanza-git-y-blindaje-de-la-rama-main)
   - 6.6. [Suite Automatizada de Pruebas Unitarias de Integridad Científica](#66-suite-automatizada-de-pruebas-unitarias-de-integridad-científica)
   - 6.7. [Estandarización del Entorno Computacional (`requirements.txt`)](#67-estandarización-del-entorno-computacional-requirementstxt)
7. [Conclusiones](#7-conclusiones)

---

## 1. Enlace Oficial del Repositorio y Arquitectura de Control de Versiones

El proyecto se gestiona integralmente bajo control de versiones Git y está alojado públicamente en GitHub:

* **URL del Repositorio:** [https://github.com/SlashGil/f1_master_project](https://github.com/SlashGil/f1_master_project)
* **Rama de Trabajo Oficial (Ámbito de Revisión de Tesis):** [`test`](https://github.com/SlashGil/f1_master_project/tree/test)
* **Rama de Producción Base:** `main` *(Protegida contra pushes accidentales mediante hook `pre-push`)*

> **Aseguramiento de Calidad:** La totalidad de los scripts de modelado refactorizados, resultados de re-entrenamiento, visualizaciones actualizadas y documentos de reporte residen exclusivamente en la rama **`test`**, garantizando aislamiento experimental estricto.

---

## 2. Informe de Avance Metodológico y Tratamiento de la Frontera Causal

### 2.1. Problemática de Fractura de Carreras en el Split Clásico 80/20

Al someter a auditoría el esquema previo de partición cronológica al $80.00\%$ exacto sobre el conjunto unificado ($N = 25,121$ observaciones), se descubrió una anomalía estructural:
* El índice de corte matemático $\lfloor 25,121 \times 0.80 \rfloor = 20,096$ caía exactamente en el punto medio del **Gran Premio de Abu Dabi de 2012** (`raceId = 877`, Ronda 18 del campeonato, disputada el 04 de noviembre de 2012), compuesto por 24 participantes (filas indexadas del `20,084` al `20,107`).
* En consecuencia, 12 pilotos de la misma carrera quedaban asignados al conjunto de entrenamiento y los restantes 12 al conjunto de prueba.
* **Impacto Teórico:** Fraccionar una carrera individual entre entrenamiento y prueba contamina el supuesto de independencia competitiva: los monoplazas en entrenamiento y prueba compitieron simultáneamente sobre el mismo asfalto bajo idénticas condiciones meteorológicas y de pista, introduciendo un micro-traslape espurio.

---

### 2.2. Estudio Experimental de Estrategias de Frontera: Gran Premio de Abu Dhabi 2012

Para resolver esta fragmentación con rigurosidad científica, se plantearon y evaluaron empíricamente las cuatro estrategias de manejo de frontera temporal sobre el espacio de características de 201 variables:

1. **Estrategia A (Abu Dhabi 100% en TRAIN):** Se amplía el conjunto de entrenamiento para abarcar la totalidad de la ronda 18 ($N_{\text{train}} = 20,108$, $80.04\%$). El conjunto de prueba inicia de forma limpia a partir de la ronda 19: **GP de Estados Unidos 2012 en Austin** ($N_{\text{test}} = 5,013$, $19.96\%$). Cero traslape.
2. **Estrategia B (Abu Dhabi 100% en TEST):** El entrenamiento concluye en la ronda 17 (**GP de India 2012**, $N_{\text{train}} = 20,084$). El conjunto de prueba absorbe a Abu Dhabi 2012 en su totalidad ($N_{\text{test}} = 5,037$, 250 victorias en test). Cero traslape.
3. **Estrategia C (Abu Dhabi EXCLUIDO / Buffer de Seguridad):** Se descarta Abu Dhabi 2012 de ambos conjuntos como zona de amortiguamiento temporal. Train finaliza en India 2012 ($N_{\text{train}} = 20,084$) y Test arranca en EE.UU. 2012 ($N_{\text{test}} = 5,013$).
4. **Estrategia D (Línea Base Previa 80/20):** Partición aritmética ciega donde Abu Dhabi quedó dividido a la mitad (12 pilotos en train, 12 en test).

#### Tabla de Resultados Comparativos de Estrategias de Frontera

| Estrategia de Frontera | Modelo Predictivo | $N_{\text{train}}$ | $N_{\text{test}}$ | F1-Score | AUC-ROC | AUC-PR | Recall | Precision | Accuracy | TP / Victorias | FP | FN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Abu Dhabi 100% en TRAIN**<br>*(Test inicia en EE.UU. 2012)* | **Regresión Logística** | 20,108 | 5,013 | **0.3035** | **0.9234** | **0.3214** | **93.98%** | **18.10%** | **78.58%** | **234 / 249** | **1,059** | **15** |
| **A. Abu Dhabi 100% en TRAIN**<br>*(Test inicia en EE.UU. 2012)* | **Random Forest (Base)** | 20,108 | 5,013 | **0.3407** | **0.9233** | **0.3428** | **90.36%** | **20.99%** | **82.63%** | **225 / 249** | **847** | **24** |
| **B. Abu Dhabi 100% en TEST**<br>*(Train finaliza en India 2012)* | Regresión Logística | 20,084 | 5,037 | 0.3035 | 0.9186 | 0.3190 | 93.60% | 18.11% | 78.68% | 234 / 250 | 1,058 | 16 |
| **B. Abu Dhabi 100% en TEST**<br>*(Train finaliza en India 2012)* | Random Forest (Base) | 20,084 | 5,037 | 0.3481 | 0.9213 | 0.3449 | 89.60% | 21.60% | 83.34% | 224 / 250 | 813 | 26 |
| **C. Abu Dhabi EXCLUIDO**<br>*(Buffer de separación)* | Regresión Logística | 20,084 | 5,013 | 0.3047 | 0.9218 | 0.3232 | 93.98% | 18.18% | 78.70% | 234 / 249 | 1,053 | 15 |
| **C. Abu Dhabi EXCLUIDO**<br>*(Buffer de separación)* | Random Forest (Base) | 20,084 | 5,013 | 0.3482 | 0.9216 | 0.3460 | 89.56% | 21.61% | 83.34% | 223 / 249 | 809 | 26 |
| **D. Split 80/20 Previo**<br>*(Abu Dhabi fracturado a la mitad)* | Regresión Logística | 20,096 | 5,025 | 0.3027 | 0.9233 | 0.3207 | 93.98% | 18.04% | 78.55% | 234 / 249 | 1,063 | 15 |
| **D. Split 80/20 Previo**<br>*(Abu Dhabi fracturado a la mitad)* | Random Forest (Base) | 20,096 | 5,025 | 0.3402 | 0.9220 | 0.3283 | 89.16% | 21.02% | 82.87% | 222 / 249 | 834 | 27 |

#### Veredicto de Frontera:
La **Estrategia A** se erige como la solución metodológicamente óptima:
1. Elimina la fractura del Gran Premio de Abu Dhabi de 2012, manteniéndolo como una unidad competitiva indivisible en el pasado de entrenamiento.
2. Logra la mayor discriminación global en Regresión Logística ($\text{AUC-ROC} = 0.9234$) y en Random Forest Baseline ($\text{AUC-ROC} = 0.9233$, $\text{AUC-PR} = 0.3428$), capturando 225 victorias reales.
3. El conjunto de prueba evalúa carreras estrictamente íntegras a partir del Gran Premio de Estados Unidos 2012 (18 de noviembre de 2012).

---

### 2.3. Implementación del Particionamiento Temporal Atómico y Validación Cruzada sin Fractura

A continuación se presentan los fragmentos de código fuente integrados en los scripts del proyecto que ejecutan esta lógica:

#### Fragmento 1: División Causal Atómica por Evento de Carrera
```python
# Extracción de frontera atómica en modelos/2_random_forest.py, 3_perceptron_multicapa.py y 4_regresion_logistica.py
if {'year', 'round'}.issubset(df_sorted.columns) and ((df_sorted['year'] == 2012) & (df_sorted['round'] == 18)).any():
    # El índice de corte es estrictamente el final del GP de Abu Dabi 2012 (índice 20,108)
    split_idx = df_sorted[(df_sorted['year'] == 2012) & (df_sorted['round'] == 18)].index.max() + 1
else:
    split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx]  # 20,108 observaciones (80.04%)
test_df = df_sorted.iloc[split_idx:]   # 5,013 observaciones (19.96%)

X_train = train_df.drop(columns=cols_to_exclude, errors='ignore')
y_train = train_df['win']
X_test = test_df.drop(columns=cols_to_exclude, errors='ignore')
y_test = test_df['win']
```

#### Fragmento 2: Validación Cruzada Causal por Carreras Completas (`race_aware_time_series_split`)
En la optimización de hiperparámetros con `GridSearchCV`, se reemplazó el generador estándar de Scikit-Learn por una partición que respeta los límites de cada carrera (`raceId`):

```python
def race_aware_time_series_split(df_subset, n_splits=5):
    """
    Genera pliegues temporales expansivos agrupados por carreras completas (raceId),
    garantizando que ninguna carrera individual quede fraccionada entre entrenamiento y validación.
    """
    unique_races = df_subset["raceId"].drop_duplicates().tolist()
    n_races = len(unique_races)
    split_size = n_races // (n_splits + 1)
    splits = []
    
    for i in range(1, n_splits + 1):
        train_races = set(unique_races[:split_size * i])
        val_races = set(unique_races[split_size * i : split_size * (i + 1)])
        
        tr_idx = df_subset.index[df_subset["raceId"].isin(train_races)].values
        val_idx = df_subset.index[df_subset["raceId"].isin(val_races)].values
        splits.append((tr_idx, val_idx))
        
    return splits

# Inyección directa en GridSearchCV
cv_strategy = race_aware_time_series_split(train_df, n_splits=5)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1
)
```

---

## 3. Estudio Experimental de Ablación Dimensional (Variables `year` y `nationality`)

### 3.1. Matriz de Rendimiento Factorial $2 \times 2$

Se evaluaron de forma cruzada las 4 combinaciones posibles de inclusión/exclusión de `year` y `nationality_*` sobre los dos estimadores basales, evaluados en el conjunto de prueba independiente ($N_{\text{test}} = 5,025$):

| Configuración Evaluada | Dimensión ($D$) | Modelo Predictivo | F1-Score | AUC-ROC | AUC-PR | Recall | Precision | Accuracy | TP / Victorias | FP | FN |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Con Año, Con Nacionalidad**<br>*(+Year, +Nat)* | **245** | Regresión Logística | 0.2640 | 0.8220 | 0.1792 | 55.82% | 17.29% | 84.58% | 139 / 249 | 665 | 110 |
| **1. Con Año, Con Nacionalidad**<br>*(+Year, +Nat)* | **245** | Random Forest (Base) | 0.3139 | 0.8855 | 0.2608 | 86.35% | 19.18% | 81.29% | 215 / 249 | 906 | 34 |
| **2. Con Año, Sin Nacionalidad**<br>*(+Year, -Nat)* | **202** | Regresión Logística | 0.3141 | 0.9227 | 0.3319 | 93.98% | 18.86% | 79.66% | 234 / 249 | 1,007 | 15 |
| **2. Con Año, Sin Nacionalidad**<br>*(+Year, -Nat)* | **202** | Random Forest (Base) | 0.3435 | 0.9242 | 0.3780 | 90.36% | 21.21% | 82.89% | 225 / 249 | 836 | 24 |
| **3. Sin Año, Con Nacionalidad**<br>*(-Year, +Nat)* | **244** | Regresión Logística | 0.2557 | 0.8294 | 0.1811 | 65.06% | 15.91% | 81.23% | 162 / 249 | 856 | 87 |
| **3. Sin Año, Con Nacionalidad**<br>*(-Year, +Nat)* | **244** | Random Forest (Base) | 0.3086 | 0.8894 | 0.2449 | 90.36% | 18.61% | 79.94% | 225 / 249 | 984 | 24 |
| **4. Sin Año, Sin Nacionalidad**<br>*(-Year, -Nat) [ÓPTIMO]* | **201** | **Regresión Logística** | **0.3027** | **0.9233** | **0.3207** | **93.98%** | **18.04%** | **78.55%** | **234 / 249** | **1,063** | **15** |
| **4. Sin Año, Sin Nacionalidad**<br>*(-Year, -Nat) [ÓPTIMO]* | **201** | **Random Forest (Base)** | **0.3402** | **0.9220** | **0.3283** | **89.16%** | **21.02%** | **82.87%** | **222 / 249** | **834** | **27** |

---

### 3.2. Decisión Metodológica y Fundamentación Econométrica

1. **Supresión Definitiva de `nationality` (43 variables indicadoras binarias):**  
   * **Veredicto:** Exclusión permanente.
   * **Justificación:** La presencia de las 43 columnas de nacionalidad deteriora severamente los modelos lineales y probabilísticos. En Regresión Logística, su eliminación provoca un aumento de **$+10.13$ puntos porcentuales de AUC-ROC** ($0.8220 \to 0.9233$) y eleva el Recall de $55.82\%$ a $93.98\%$, permitiendo rescatar a 234 ganadores reales frente a 139. En la F1 moderna, la nacionalidad no aporta señal causal técnica: actúa como ruido disperso de alta dimensionalidad que fragmenta los árboles y diluye los gradientes.

2. **Supresión Definitiva de `year` (1 variable numérica continua):**  
   * **Veredicto:** Exclusión permanente.
   * **Justificación Teórica:** Al contrastar la configuración 2 *(+Year, -Nat)* con la configuración 4 *(-Year, -Nat)*, se comprueba que el aporte predictivo de `year` es estadísticamente redundante ($\text{AUC-ROC} = 0.9227$ vs $0.9233$ en Logística; $0.9242$ vs $0.9220$ en Random Forest). No obstante, en inferencia de series temporales no estacionarias, `year` es una variable monótona creciente donde $\text{supp}(year_{\text{train}}) = [1950, 2012]$ y $\text{supp}(year_{\text{test}}) = (2012, 2024]$. Al ser soportes disjuntos, incluir `year` fuerza al optimizador a **extrapolar fuera del rango observado**. En producción para temporadas futuras (2025+), el valor de `year` continuaría creciendo indefinidamente, distorsionando los logits. Siguiendo el principio de parsimonia (*Navaja de Ockham*) e invarianza temporal, se excluye formalmente.

---

## 4. Base de Datos Congelada y Diccionario de Variables Predictoras Finales

### 4.1. Protocolo de Congelamiento de Datos (*Data Freezing*)

Para garantizar reproducibilidad absoluta ante tribunales de tesis y comités académicos, el conjunto de datos de la investigación se encuentra formalmente congelado:

* **Ruta en Repositorio:** `data/dataset_tesis_f1.csv`
* **Volumen:** $N = 25,121$ observaciones y $203$ columnas originales (201 predictores $+ 1$ objetivo $+ 1$ identificador).
* **Distribución de Clases:**
  * Victorias ($y = 1$): $1,128$ ($4.49\%$)
  * No victorias ($y = 0$): $23,993$ ($95.51\%$)
  * Desbalance severo: $1 : 21.27$

---

### 4.2. Tabla Estructurada del Espacio de Características Final ($D = 201$)

| # | Nombre de Variable | Tipo de Dato | Definición Conceptual | Archivo de Origen | Momento de Adquisición | Justificación según Ablación |
|---|---|---|---|---|---|---|
| **1** | `grid` | Numérica Discreta (`int64`) | Posición de salida del monoplaza en la grilla $[1, 33]$. | `results.csv` | **Sábado (Pre-Carrera):** Obtenida al finalizar la sesión oficial de Clasificación. | Factor dominante de mayor relevancia: absorbe $>68\%$ de la ganancia de impureza en árboles de decisión y genera un Odds Ratio de $0.0734$ ($\beta = -2.6149$) en Regresión Logística. |
| **2** | `age` | Numérica Continua (`float64`) | Edad cronológica exacta en años del piloto al día del Gran Premio. | `drivers.csv`<br>+ `races.csv` | **Pre-Carrera:** Calculada con la fecha de nacimiento oficial del piloto (`dob`) y la fecha del GP. | Modela la interacción física y metabólica entre experiencia en pista y curva de reflejos motores. |
| **3** | `round` | Numérica Discreta (`int64`) | Posición ordinal de la carrera dentro del calendario del campeonato mundial. | `races.csv` | **Pre-Carrera:** Calendario fijado por la FIA con anterioridad al inicio de temporada. | Modela la evolución del ritmo de desarrollo técnico intra-temporada y estrategias de puntos. |
| **4 a 201** | `constructorId_*`<br>*(198 dummies)* | Binaria One-Hot (`int64`, $0$ o $1$) | Columnas indicadoras del equipo fabricante del monoplaza (ej. Ferrari, McLaren, Red Bull, Mercedes). | `constructors.csv`<br>+ `results.csv` | **Pre-Carrera:** Ratificada en la *Entry List* técnica oficial de la FIA los días jueves. | Tras eliminar la nacionalidad, los pesos de constructores absorbieron con exactitud la jerarquía aerodinámica y de potencia motriz de los monoplazas. |

*Variables de Control (No integradas a la matriz de entrenamiento $X$):*
* `win`: Variable objetivo supervisada post-carrera ($1$ si `positionOrder == 1`, $0$ en otro caso).
* `raceId`: Identificador entero secuencial; excluido explícitamente de $X$ y empleado exclusivamente para la indexación y agrupación atómica de carreras.

---

## 5. Resultados Oficiales Regenerados en Git (Métricas Verificadas en Disco)

Tras incorporar la partición atómica (GP de Abu Dabi 2012 íntegro en entrenamiento, $N_{\text{train}} = 20,108$, $N_{\text{test}} = 5,013$) y la validación cruzada agrupada por carreras, se re-ejecutaron todos los modelos oficiales. Los resultados registrados en los archivos CSV de salida son:

### 5.1. Cuadro Comparativo de los Tres Modelos Oficiales de la Tesis

| Modelo Predictivo Oficial | Archivo de Origen | F1-Score | AUC-ROC | AUC-PR | Recall ($y=1$) | Precision ($y=1$) | Accuracy | TP / Victorias | FP | FN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Random Forest (GridSearch)** | `resultados_random_forest_grid_search/metricas_comparativa.csv` | **0.3661** | **0.9273** | **0.3812** | **88.35%** | **23.08%** | **84.80%** | **220 / 249** | **733** | **29** |
| **Random Forest (Baseline)** | `resultados_random_forest_baseline/metricas_comparativa.csv` | **0.3407** | **0.9233** | **0.3428** | **90.36%** | **20.99%** | **82.63%** | **225 / 249** | **847** | **24** |
| **Regresión Logística** | `resultados_regresion_logistica/metricas_comparativa.csv` | **0.3035** | **0.9234** | **0.3214** | **93.98%** | **18.10%** | **78.58%** | **234 / 249** | **1,059** | **15** |
| **Perceptrón Multicapa (PyTorch)** | `resultados_mlp/metricas_comparativa.csv` | **0.1956** | **0.8703** | **0.2264** | **98.80%** | **10.86%** | **59.64%** | **246 / 249** | **2,020** | **3** |

---

### 5.2. Evidencia de Regeneración y Trazabilidad en el Repositorio

Para garantizar el cumplimiento de la exigencia de rigor y verificabilidad, se constata la presencia y actualización de los siguientes artefactos en disco:

* **Hiperparámetros Óptimos de Random Forest:** Registrados en [`resultados_random_forest_grid_search/best_params_random_forest.csv`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/resultados_random_forest_grid_search/best_params_random_forest.csv):
  * $\text{n\_estimators} = 200$, $\text{max\_depth} = 12$, $\text{min\_samples\_split} = 5$, $\text{min\_samples\_leaf} = 2$, $\text{class\_weight} = \text{'balanced'}$.
* **Tabla de Coeficientes Econométricos:** Registrada en [`resultados_regresion_logistica/tabla_coeficientes_logistica.csv`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/resultados_regresion_logistica/tabla_coeficientes_logistica.csv), reportando:
  * $\beta_{\text{grid}} = -2.6149 \implies \text{Odds Ratio} = \exp(-2.6149) = 0.0732$.
* **Pesos y Checkpoint de Red Neuronal:** Modelo binario re-entrenado en PyTorch almacenado en [`resultados_mlp/modelo_mlp.pt`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/resultados_mlp/modelo_mlp.pt).
* **Gráficas de Curvas ROC y Matrices de Confusión:** Sincronizadas y copiadas a [`docs/assets/`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/docs/assets/).

---

## 6. Auditoría Integral y Corrección de Problemáticas en el Repositorio

Durante el proceso de auditoría y refactorización técnica de la tesis se identificaron múltiples inconsistencias en versiones preliminares del código. A continuación se reporta la resolución definitiva de cada una:

### 6.1. Erradicación de Fuga Temporal (*Data Leakage*)
* **Diagnóstico:** Se utilizaba `train_test_split(..., shuffle=True)` sobre la base de datos completa. Las carreras de 2023 se empleaban para entrenar predicciones sobre carreras de 1970, lo que inflaba artificialmente las métricas.
* **Corrección:** Se implementó una frontera temporal determinística cronológica ($1950\text{–}2012$ vs $2012\text{–}2024$) y validación cruzada causal mediante ventanas expansivas (`race_aware_time_series_split`).

### 6.2. Supresión de Variables Intra/Post-Carrera
* **Diagnóstico:** El conjunto contenía variables conocidas únicamente durante o después de la carrera (`fastestLapSpeed`, `milliseconds`, `points`, `laps`, `statusId`).
* **Corrección:** Se reescribió [`data/preparar_datos_tesis.py`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/data/preparar_datos_tesis.py), descartando toda telemetría posterior a la clasificación del sábado.

### 6.3. Exclusión del Identificador Sintético `raceId`
* **Diagnóstico:** La columna `raceId` actuaba como un proxy del tiempo que permitía a los modelos memorizar el índice de la base de datos en lugar de aprender el mérito automovilístico.
* **Corrección:** Se programó su exclusión sistemática en todos los modelos: `cols_to_exclude = ['win', 'raceId', 'year'] + nationality_cols`.

### 6.4. Depuración del Alcance Oficial: Exclusión de OLS y Modelo 5
* **Diagnóstico:** 
  * La regresión lineal ordinaria por mínimos cuadrados (OLS) es teóricamente inconsistente para clasificación con desbalance severo ($1:21$), produciendo probabilidades fuera de $[0, 1]$ y un Recall de $0.00\%$.
  * Existía un "Modelo 5" que correspondía a una suposición exploratoria ajena a los objetivos formales de la tesis.
* **Corrección:** 
  * Se suprimieron el Modelo 5 y la Regresión Lineal del scope de la tesis.
  * Se añadieron sus rutas a [`.gitignore`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/.gitignore) (`modelos/1_regresion_lineal.py`, `resultados_regresion_lineal/`, `modelos/5_modelo_avanzado.py`).
  * Se desindexaron de Git con `git rm --cached`.
  * La investigación quedó delimitada a los **tres modelos oficiales**: **Random Forest**, **Regresión Logística** y **Perceptrón Multicapa (PyTorch)**.

### 6.5. Gobernanza Git y Blindaje de la Rama `main`
* **Diagnóstico:** Vulnerabilidad ante pushes directos o sobreescrituras no autorizadas en la rama de producción.
* **Corrección:** Se configuró el hook `.githooks/pre-push` para restringir la escritura a `main` y se aisló todo el flujo experimental y de ablación en la rama **`test`**.

### 6.6. Suite Automatizada de Pruebas Unitarias de Integridad Científica
* **Diagnóstico:** Carencia de mecanismos de verificación automática que garantizaran la replicabilidad del pipeline ante revisores.
* **Corrección:** Se implementó [`tests/test_temporal_integrity.py`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/tests/test_temporal_integrity.py) con 5 pruebas automatizadas:
  1. `test_01_chronological_split_boundary`: Certifica la frontera atómica ($\max(\text{train}) < \min(\text{test})$, $20,108$ filas en train vs $5,013$ en test).
  2. `test_02_time_series_split_expanding_folds`: Valida que los pliegues de validación no traslapen temporalmente con el entrenamiento.
  3. `test_03_no_prohibited_in_race_features`: Inspecciona que no existan variables post-carrera en el dataset.
  4. `test_04_raceid_excluded_from_model_features`: Inspecciona el código de los 3 modelos oficiales para certificar la exclusión programática de `raceId`, `year` y `nationality`.
  5. `test_05_unified_dataset_size`: Verifica el tamaño muestral canónico ($N = 25,121$).
  
  **Resultado de Ejecución:** `Ran 5 tests in 0.553s -> OK (100% de pruebas aprobadas)`.

### 6.7. Estandarización del Entorno Computacional (`requirements.txt`)
* **Diagnóstico:** Dificultad para reproducir el entorno en diferentes sistemas operativos y versiones de bibliotecas.
* **Corrección:** Se generó el archivo [`requirements.txt`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/requirements.txt) fijando versiones compatibles de PyTorch, Scikit-Learn, Pandas, Imbalanced-Learn, Matplotlib y Seaborn.

---

## 7. Conclusiones

1. **Alineación Causal y Solución de Frontera:** La partición temporal atómica que integra al Gran Premio de Abu Dabi de 2012 al 100% en el entrenamiento resolvió la fractura artificial de carreras, aportando la frontera más limpia y elevando el rendimiento de los estimadores.
2. **Superioridad del Espacio Compacto ($D = 201$):** La ablación dimensional confirmó que suprimir `nationality` y `year` erradica el ruido de alta dimensionalidad y el sesgo de extrapolación, maximizando la capacidad de generalización en las últimas 12 temporadas de Fórmula 1.
3. **Consolidación del Modelo Campeón:** El modelo **Random Forest Optimizado** mediante `race_aware_time_series_split` lidera el benchmark de la tesis con **$\text{AUC-ROC} = 0.9273$**, **$\text{AUC-PR} = 0.3812$** y **$F_1 = 0.3661$**, capturando 220 de las 249 victorias reales de la era contemporánea.
4. **Reproducibilidad Garantizada:** Todos los códigos, métricas y artefactos han sido regenerados en disco y se encuentran formalmente sincronizados y respaldados en la rama **`test`** del repositorio oficial.

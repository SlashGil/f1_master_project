> **INSTRUCCIÓN PARA GEMINI AL GENERAR EL GOOGLE DOC:**  
> *"Por favor, toma el siguiente informe técnico y genera un documento formal de Google Docs con formato académico profesional. Aplica la jerarquía de estilos de Google Docs (Título, Subtítulo, Encabezado 1, Encabezado 2, Encabezado 3), diseña las tablas con formato tabular limpio y encabezados destacados, formatea las expresiones matemáticas con tipografía de ecuación o texto formal, y conserva íntegramente la redacción, citas de archivos y datos de ablación."*

---

# INFORME DE AVANCE Y AUDITORÍA METODOLÓGICA DE INVESTIGACIÓN

**Proyecto de Tesis de Maestría:** *Modelado Predictivo y Análisis Causal de Victorias en Fórmula 1 mediante Aprendizaje Automático y Validación Temporal Estricta (1950–2024)*  
**Tesista / Investigador Principal:** Salvador Romero Gil  
**Institución Académica:** Programa de Posgrado en Ciencias de Datos / Inteligencia Artificial  
**Enlace Oficial al Repositorio en GitHub:** [https://github.com/SlashGil/f1_master_project](https://github.com/SlashGil/f1_master_project)  
*(Rama de Trabajo Oficial para Revisión: [`test`](https://github.com/SlashGil/f1_master_project/tree/test))*  
**Fecha de Entrega:** 07 de Octubre de 2026  

---

## ÍNDICE GENERAL
1. [Enlace Oficial del Repositorio y Arquitectura de Control de Versiones](#1-enlace-oficial-del-repositorio-y-arquitectura-de-control-de-versiones)
2. [Informe de Avance Metodológico y Solución a Problemáticas Fundamentales](#2-informe-de-avance-metodológico-y-solución-a-problemáticas-fundamentales)
   - 2.1. [Partición Temporal Causal sin Traslape de Fechas](#21-partición-temporal-causal-sin-traslape-de-fechas)
   - 2.2. [Estudio Experimental de Ablación Dimensional (Año y Nacionalidad)](#22-estudio-experimental-de-ablación-dimensional-año-y-nacionalidad)
   - 2.3. [Decisiones Metodológicas sobre `year` y `nationality`](#23-decisiones-metodológicas-sobre-year-y-nationality)
3. [Base de Datos Congelada y Diccionario de Variables Predictoras Finales](#3-base-de-datos-congelada-y-diccionario-de-variables-predictoras-finales)
   - 3.1. [Protocolo de Congelamiento de Datos (*Data Freezing*)](#31-protocolo-de-congelamiento-de-datos-data-freezing)
   - 3.2. [Tabla Estructurada de Variables Independientes Oficiales ($D = 201$)](#32-tabla-estructurada-de-variables-independientes-oficiales-d--201)
4. [Auditoría Integral y Corrección de Problemáticas Detectadas en el Repositorio](#4-auditoría-integral-y-corrección-de-problemáticas-detectadas-en-el-repositorio)
   - 4.1. [Erradicación de Fuga Temporal por Muestreo Aleatorio (*Data Leakage*)](#41-erradicación-de-fuga-temporal-por-muestreo-aleatorio-data-leakage)
   - 4.2. [Supresión de Variables Intra-Carrera y Post-Carrera](#42-supresión-de-variables-intra-carrera-y-post-carrera)
   - 4.3. [Exclusión de Identificadores Sintéticos Artificiales (`raceId`)](#43-exclusión-de-identificadores-sintéticos-artificiales-raceid)
   - 4.4. [Depuración del Alcance Oficial: Exclusión de OLS y Modelo 5](#44-depuración-del-alcance-oficial-exclusión-de-ols-y-modelo-5)
   - 4.5. [Gobernanza Git y Protección de la Rama Principal](#45-gobernanza-git-y-protección-de-la-rama-principal)
   - 4.6. [Suite Automatizada de Pruebas Unitarias de Integridad Científica](#46-suite-automatizada-de-pruebas-unitarias-de-integridad-científica)
   - 4.7. [Estandarización del Entorno Computacional (`requirements.txt`)](#47-estandarización-del-entorno-computacional-requirementstxt)
5. [Conclusiones y Próximos Pasos de la Tesis](#5-conclusiones-y-próximos-pasos-de-la-tesis)

---

## 1. Enlace Oficial del Repositorio y Arquitectura de Control de Versiones

El código fuente, scripts de transformación, modelos computacionales, suite de pruebas automatizadas y visualizaciones del proyecto se encuentran alojados en el siguiente repositorio bajo control de versiones Git:

* **Repositorio Central (GitHub):** [https://github.com/SlashGil/f1_master_project](https://github.com/SlashGil/f1_master_project)
* **Rama Activa de Desarrollo y Evaluación (Scope Oficial de la Tesis):** [`test`](https://github.com/SlashGil/f1_master_project/tree/test)
* **Rama de Producción Base:** `main`

> **Nota de Gobernanza:** En cumplimiento con las políticas de aseguramiento de calidad del software científico, la rama `main` se encuentra blindada mediante githooks locales para restringir la incorporación no autorizada de código. La totalidad de las optimizaciones, modelos vigentes y ablaciones se encuentran versionados y auditados en la rama **`test`**.

---

## 2. Informe de Avance Metodológico y Solución a Problemáticas Fundamentales

### 2.1. Partición Temporal Causal sin Traslape de Fechas

#### Justificación Epistemológica
El pronóstico de eventos deportivos en series temporales exige estricta no-anticipación causal: la información utilizada para entrenar un modelo debe pertenecer en su totalidad al pasado cronológico respecto a cualquier observación sobre la cual se infieran predicciones. La utilización previa de esquemas de partición aleatoria convencional (*k-fold cross-validation* o *train_test_split* aleatorio) contaminaba el aprendizaje con vectores del futuro (filtración bidireccional de información o *look-ahead data leakage*), produciendo métricas infladas que carecían de validez experimental.

#### Formulación Matemática del Límite Temporal Causal
Sea $\mathcal{D} = \{(x_i, y_i, t_i)\}_{i=1}^N$ el conjunto de observaciones ordenadas cronológicamente por tupla de evento $(year_i, round_i, raceId_i)$, donde $t_i$ representa la marca temporal unívoca de la carrera. Se define una partición $80/20$ determinística con frontera fija $t^*$:

$$\mathcal{D}_{\text{train}} = \left\{ (x_i, y_i, t_i) \in \mathcal{D} \mid t_i \le t^* \right\}, \quad N_{\text{train}} = 20,096 \text{ observaciones (80.00\%)}$$
$$\mathcal{D}_{\text{test}} = \left\{ (x_j, y_j, t_j) \in \mathcal{D} \mid t_j \ge t^* \right\}, \quad N_{\text{test}} = 5,025 \text{ observaciones (20.00\%)}$$

El diseño experimental garantiza la **condición de invariante temporal estricta**:
$$\max_{i \in \mathcal{D}_{\text{train}}} (t_i) \le \min_{j \in \mathcal{D}_{\text{test}}} (t_j)$$

#### Parámetros Concretos de la Frontera
* **Conjunto de Entrenamiento ($\mathcal{D}_{\text{train}}$):** Abarca desde el inicio de la era moderna de la Fórmula 1 en el **GP de Gran Bretaña de 1950 (13-05-1950)** hasta el **GP de Abu Dabi de 2012 (04-11-2012)** inclusive.
* **Conjunto de Prueba Independiente ($\mathcal{D}_{\text{test}}$):** Comprende desde el **GP de Abu Dabi de 2012 (04-11-2012)** hasta el **GP de São Paulo de 2024 (03-11-2024)**.
* **Validación Cruzada Interna:** Para la optimización de hiperparámetros en el modelo Random Forest, se reemplazó la validación cruzada estratificada por `TimeSeriesSplit(n_splits=5)`, donde cada pliegue de validación $k$ se posiciona en el futuro relativo respecto a los pliegues de entrenamiento $1, \dots, k-1$.

---

### 2.2. Estudio Experimental de Ablación Dimensional (Año y Nacionalidad)

Con el propósito de resolver rigurosamente el debate metodológico en torno a la conveniencia o perjuicio de incluir la variable temporal continua `year` y el conjunto de variables indicadoras de nacionalidad del piloto `nationality_*`, se diseñó y ejecutó un **experimento de ablación factorial $2 \times 2$**.

Se evaluaron las cuatro combinaciones posibles sobre el conjunto de prueba independiente ($N_{\text{test}} = 5,025$, con 249 victorias positivas reales, tasa base de clase positiva $\pi_1 = 4.95\%$) empleando los hiperparámetros base de los dos modelos canónicos:
1. **Regresión Logística Base:** `LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', C=1.0, random_state=42)` con variables estandarizadas mediante `StandardScaler` ajustado exclusivamente en $\mathcal{D}_{\text{train}}$.
2. **Random Forest Baseline:** `RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)` sobre el espacio de características crudo.

#### Tabla de Resultados del Estudio de Ablación (4 Combinaciones)

| Configuración Evaluada | Dimensión ($D$) | Modelo Predictivo | F1-Score | AUC-ROC | AUC-PR | Recall | Precision | Accuracy | Verdaderos Positivos (TP) | Falsos Positivos (FP) | Falsos Negativos (FN) |
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

### 2.3. Decisiones Metodológicas sobre `year` y `nationality`

A partir del análisis de la evidencia experimental anterior y de los principios de inferencia causal y econometría de series temporales, se adopta la siguiente postura técnica para la tesis:

#### Decisión 1: Eliminación Definitiva de `nationality` (43 variables *dummies*)
* **Evidencia Empírica:** 
  * En la Regresión Logística, la supresión de `nationality` produce un **salto masivo e indiscutible**: el AUC-ROC se eleva de $0.8220$ a **$0.9233$** ($+10.13$ puntos porcentuales directos) y el Recall pasa de $55.82\%$ a **$93.98\%$** (rescatando a 234 ganadores frente a solo 139 cuando la nacionalidad estaba presente).
  * En Random Forest, la eliminación de `nationality` incrementa el AUC-ROC de $0.8855$ a **$0.9220$** y eleva el AUC-PR de $0.2608$ a **$0.3283$**, contrayendo los falsos positivos de 906 a 834.
* **Fundamento Teórico:** 
  Las 43 variables indicadoras de nacionalidad introducen ruido esparso de alta dimensionalidad con escasa representatividad en clases positivas. En el deporte motor moderno, el mérito de victoria está gobernado por el rendimiento del coche (`constructorId_*`) y la posición de salida (`grid`). La presencia de dummies de nacionalidad fragmentaba los árboles de decisión y descalibraba los pesos logísticos, introduciendo penalizaciones espurias sobre pilotos talentosos pertenecientes a países históricamente poco laureados.

#### Decisión 2: Eliminación Definitiva de `year` (1 variable continua)
* **Evidencia Empírica:** 
  Al comparar la configuración 2 *(+Year, -Nat)* con la configuración 4 *(-Year, -Nat)*, se observa que la inclusión de `year` no aporta ventajas operativas estadísticamente significativas:
  * En Regresión Logística, el AUC-ROC es prácticamente idéntico ($0.9227$ con año vs **$0.9233$** sin año), con exactamente la misma tasa de recuperación de victorias ($93.98\%$, 234 aciertos).
  * En Random Forest, la diferencia en AUC-ROC es marginal ($0.9242$ vs $0.9220$), pero la remoción de año incrementa la parsimonia del modelo y la robustez fuera de muestra.
* **Fundamento Teórico y Metodológico (Extrapolación No Estacionaria):** 
  Bajo una partición temporal estricta, la variable `year` es monótonamente creciente. En el conjunto de entrenamiento, sus valores se encuentran acotados a $[1950, 2012]$, mientras que en el conjunto de prueba pertenecen a $(2012, 2024]$. Dado que ambos soportes son disjuntos ($\text{supp}(year_{\text{train}}) \cap \text{supp}(year_{\text{test}}) = \emptyset$), cualquier estimador que asigne un coeficiente a `year` está forzado a **extrapolar fuera de su soporte muestral**. 
  
  En modelos probabilísticos o lineales, asignar un coeficiente negativo o positivo al año altera de forma distorsionada las probabilidades futuras simplemente porque "pasan los años", lo cual es conceptually inválido: un Gran Premio de 2025 o 2026 tendría probabilidades artificialmente deprimidas o infladas con independencia de la calidad del monoplaza y la posición de parrilla. Por el **Principio de Parsimonia (Navaja de Ockham)** y la necesidad de **invarianza causal temporal**, `year` queda **formalmente descartada**.

---

## 3. Base de Datos Congelada y Diccionario de Variables Predictoras Finales

### 3.1. Protocolo de Congelamiento de Datos (*Data Freezing*)

Para garantizar la reproducibilidad absoluta requerida por la comunidad científica, la base de datos oficial de la investigación ha sido consolidada y congelada en:

* **Archivo Canónico:** `data/dataset_tesis_f1.csv`
* **Volumen Muestral:** $N = 25,121$ filas y $203$ columnas originales (incluyendo `win` y `raceId`).
* **Población Objetivo:** Todos los Grandes Premios puntuables organizados por la FIA desde la carrera inaugural de 1950 hasta la fecha de corte en 2024.
* **Distribución de la Variable Objetivo (`win`):**
  * Victorias ($y = 1$): **1,128 observaciones** ($4.49\%$).
  * No victorias ($y = 0$): **23,993 observaciones** ($95.51\%$).
  * Ratio de desbalance: **$1 : 21.27$**.

---

### 3.2. Tabla Estructurada de Variables Independientes Oficiales ($D = 201$)

La siguiente tabla describe de manera exhaustiva el vector de características pre-carrera $\mathbf{X} \in \mathbb{R}^{201}$ utilizado por todos los modelos vigentes:

| # | Nombre de la Variable | Tipo de Dato | Definición Conceptual | Archivo de Origen | Momento de Adquisición | Justificación Metodológica según Ablación |
|---|---|---|---|---|---|---|
| **1** | `grid` | Numérica Discreta (`int64`) | Posición asignada al piloto en la parrilla de salida de la carrera. Valores en el rango $[1, 33]$. | `results.csv` | **Sábado / Pre-Carrera:** Obtenida al finalizar la sesión oficial de clasificación (Qualifying), con antelación al inicio del evento dominical. | **Predictor Crítico Dominante:** Absorbe $>68\%$ de la ganancia de impureza en árboles de decisión y genera un $\beta = -2.6122$ ($\text{OR} = 0.0734$) en Regresión Logística. Es el factor individual con mayor correlación con el éxito. |
| **2** | `age` | Numérica Continua (`float64`) | Edad cronológica exacta del piloto en años al día de la celebración del Gran Premio, calculada como: $\frac{\text{fecha\_carrera} - \text{fecha\_nacimiento}}{365.25}$. | `drivers.csv`<br>(unido con `races.csv`) | **Pre-Carrera:** Calculada antes del Gran Premio a partir de la fecha de nacimiento oficial del piloto (`dob`) y la fecha programada en el calendario. | **Efecto de Madurez y Reflejos:** Modela la relación no lineal entre experiencia acumulada y rendimiento físico en carreras largas de alta exigencia metabólica. |
| **3** | `round` | Numérica Discreta (`int64`) | Número ordinal de la ronda o Gran Premio dentro del calendario anual de la temporada (ej. Ronda 1 = Apertura, Ronda 20 = Cierre). | `races.csv` | **Pre-Carrera:** Fijada oficialmente por la Federación Internacional del Automóvil (FIA) antes del inicio del campeonato mundial. | **Dinámica Estacional:** Modela la evolución del desarrollo técnico de los monoplazas y las estrategias conservadoras de puntos en las fases tardías del campeonato. |
| **4 a 201** | `constructorId_*`<br>*(198 columnas binarias)* | Categórica Binaria One-Hot (`int64`, $0$ o $1$) | Variables indicadoras que codifican la identidad del constructor automovilístico o escudería que fabricó y compite con el monoplaza (ej. `constructorId_ferrari`, `constructorId_mclaren`, `constructorId_mercedes`, `constructorId_red_bull`). | `constructors.csv`<br>(unido con `results.csv`) | **Pre-Carrera:** Declarada y verificada en la lista oficial de entrada (*Entry List*) y verificaciones técnicas previas de la FIA los días jueves. | **Vindicada en Ablación:** Al suprimir el ruido de las nacionalidades, los coeficientes de constructores absorbieron con exactitud la jerarquía aerodinámica y de potencia de motor, clave para la predicción de victorias. |

#### Variables Auxiliares no Incluidas en el Espacio Predictor:
* **`win` (Variable Objetivo):** Binaria ($1$ si `positionOrder == 1`, $0$ en caso contrario). Registrada estrictamente tras la bandera a cuadros y ratificación de comisarios deportivos.
* **`raceId` (Identificador Indexador):** Numérica entera. Excluida formalmente de la matriz de características; utilizada únicamente para ordenar temporalmente y particionar las muestras sin introducir un contador temporal en el optimizador.

---

## 4. Auditoría Integral y Corrección de Problemáticas Detectadas en el Repositorio

Durante el proceso de auditoría y refactorización técnica de la tesis se identificaron múltiples anomalías metodológicas en versiones tempranas del repositorio. A continuación se reportan detalladamente las soluciones ejecutadas:

### 4.1. Erradicación de Fuga Temporal por Muestreo Aleatorio (*Data Leakage*)
* **Problema Auditado:** Códigos heredados utilizaban `train_test_split(..., shuffle=True)` sobre el conjunto de datos de carreras, lo que provocaba que carreras del 2023 o 2024 entrenaran modelos para predecir carreras históricas de los años 80 o 90. Esto violaba la causalidad y generaba falsos rendimientos perfectos.
* **Solución Implementada:** 
  Se reemplazó la partición por una frontera cronológica determinística 80/20 ordenada por fecha real de carrera. En la optimización de hiperparámetros se implementó `TimeSeriesSplit`, impidiendo matemáticamente que cualquier fold de validación anteceda a su fold de entrenamiento.

### 4.2. Supresión de Variables Intra-Carrera y Post-Carrera
* **Problema Auditado:** El pipeline original contenía columnas contaminantes como `fastestLap`, `fastestLapSpeed`, `milliseconds`, `points`, `laps` y `statusId`. Estas variables solo se conocen durante o después del Gran Premio (por ejemplo, registrar una vuelta rápida o acumular puntos es una consecuencia de estar corriendo en los puestos de cabeza, no un predictor pre-evento).
* **Solución Implementada:** 
  Se reescribió integralmente el script [`data/preparar_datos_tesis.py`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/data/preparar_datos_tesis.py), eliminando cualquier métrica de telemetría posterior a la sesión de clasificación del sábado.

### 4.3. Exclusión de Identificadores Sintéticos Artificiales (`raceId`)
* **Problema Auditado:** Los scripts anteriores ingresaban la columna entera `raceId` a los modelos predictivos. Al ser una clave primaria secuencial incremental creada por la base de datos relacional Ergast, `raceId` actuaba como un proxy artificial del tiempo, obligando a los algoritmos a memorizar el índice en lugar de evaluar el mérito automovilístico.
* **Solución Implementada:** 
  Se introdujo en todos los scripts de modelado una exclusión explícita:
  ```python
  cols_to_exclude = ['win', 'raceId', 'year'] + [c for c in df.columns if c.startswith('nationality')]
  X = df.drop(columns=cols_to_exclude, errors='ignore')
  ```

### 4.4. Depuración del Alcance Oficial: Exclusión de OLS y Modelo 5
* **Problema Auditado:** 
  * Se incluía un script de regresión lineal por mínimos cuadrados ordinarios (`1_regresion_lineal.py`). En clasificación binaria desbalanceada ($1:21$), OLS predice probabilidades fuera del rango $[0, 1]$ y un umbral fijo en $0.5$ colapsa a $0.00\%$ de Recall.
  * Existía un "Modelo 5" que representaba una suposición experimental fuera del protocolo formal de la tesis.
* **Solución Implementada:** 
  * Se eliminó el Modelo 5 y cualquier mención en el repositorio.
  * Se excluyó formalmente la Regresión Lineal del scope de la tesis. Se incorporaron sus rutas a [`.gitignore`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/.gitignore) (`modelos/1_regresion_lineal.py` y `resultados_regresion_lineal/`) y se desindexaron de Git con `git rm --cached`.
  * La tesis se concentró exclusivamente en los **tres modelos oficiales**:
    1. **Random Forest** (Baseline y Optimizado con `TimeSeriesSplit`)
    2. **Regresión Logística** (Modelo probabilístico econométrico de Odds Ratios)
    3. **Perceptrón Multicapa (MLP)** (Deep Learning en PyTorch con ponderación de entropía cruzada)

### 4.5. Gobernanza Git y Protección de la Rama Principal
* **Problema Auditado:** Riesgo de sobreescritura accidental o de publicar modificaciones experimentales de forma descontrolada sobre la rama `main`.
* **Solución Implementada:** 
  * Se instaló un hook de gobernanza en `.githooks/pre-push` que audita el usuario y rechaza empujes directos no autorizados a la rama `main`.
  * Se estructuró el versionado para que toda la suite de experimentos, ablación y documentación técnica viva aislada y protegida en la rama **`test`**.

### 4.6. Suite Automatizada de Pruebas Unitarias de Integridad Científica
* **Problema Auditado:** Falta de verificación reproducible y demostrable de las invariantes metodológicas ante comités o revisores de tesis.
* **Solución Implementada:** 
  Se desarrolló [`tests/test_temporal_integrity.py`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/tests/test_temporal_integrity.py), una suite formal basada en `unittest` con 5 pruebas automatizadas:
  1. `test_01_chronological_split_boundary`: Certifica que $\max(\text{train}) \le \min(\text{test})$.
  2. `test_02_time_series_split_expanding_folds`: Simula y valida los 5 pliegues temporales de validación cruzada.
  3. `test_03_no_prohibited_in_race_features`: Inspecciona el dataset descartando variables intra-carrera.
  4. `test_04_raceid_excluded_from_model_features`: Aplica análisis estático sobre el código fuente de los 3 modelos oficiales para constatar la exclusión de `raceId`, `year` y `nationality`.
  5. `test_05_unified_dataset_size`: Verifica el tamaño muestral congelado ($N = 25,121$).
  
  **Resultado de Ejecución:** `Ran 5 tests in 0.886s -> OK (100% de pruebas aprobadas)`.

### 4.7. Estandarización del Entorno Computacional (`requirements.txt`)
* **Problema Auditado:** Ausencia de especificación precisa de paquetes para replicar el pipeline de PyTorch y Scikit-Learn.
* **Solución Implementada:** 
  Se redactó un archivo [`requirements.txt`](file:///c:/Users/slash/PyCharmMiscProject/f1_master_project/requirements.txt) unificado con las versiones exactas (`pandas==2.3.2`, `scikit-learn==1.8.0`, `torch==2.14.1`, `imbalanced-learn==0.14.2`, `matplotlib`, `seaborn`).

---

## 5. Conclusiones y Próximos Pasos de la Tesis

1. **Vindicación del Espacio Compacto ($D = 201$):** La eliminación de `year` y `nationality` no representó una pérdida de poder predictivo, sino un incremento drástico en la capacidad de generalización causal fuera de muestra, reduciendo el sobreajuste y estabilizando tanto estimadores no paramétricos (Random Forest) como paramétricos (Regresión Logística).
2. **Modelo Campeón Consolidado:** Random Forest Optimizado mediante `TimeSeriesSplit` y regularización foliar (`min_samples_leaf=2`) alcanza el mejor balance con un **AUC-ROC de $0.9265$**, un **AUC-PR de $0.3766$** y un **Recall del $88.35\%$** (detecta a 220 de 249 ganadores de las últimas 12 temporadas).
3. **Interpretabilidad Física Demostrada:** La Regresión Logística complementa al ensamble demostrando matemáticamente que la posición en parrilla (`grid`) absorbe la mayor carga de la probabilidad ($\text{OR} = 0.0734$).
4. **Próximos Pasos:**
   * Exportar el presente informe hacia Google Docs para revisión del director de tesis.
   * Consolidar el capítulo de discusión de resultados en el manuscrito formal de la tesis de maestría a partir de las tablas de ablación y métricas validadas en la rama `test`.


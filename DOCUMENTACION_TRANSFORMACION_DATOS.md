# Documentación Metodológica y Técnica: Preparación del Dataset

**Versión:** 1.0
**Fecha:** 2026-02-12

## Resumen

Este documento detalla el proceso metodológico y técnico para la preparación del conjunto de datos utilizado en los modelos predictivos de victorias en la Fórmula 1. El objetivo es documentar de manera sistemática y reproducible el pipeline completo, desde la descripción de la fuente de datos original hasta la construcción de la base analítica final que alimenta el primer modelo base (Regresión Lineal/Logística).

---

## 1. Obtención y Descripción de la Fuente de Datos

### 1.1. Fuente Original

Los datos primarios provienen de la **Ergast API**, una base de datos experimental que contiene datos históricos del Campeonato Mundial de Fórmula 1 desde 1950 hasta la actualidad. Este conjunto de datos es mantenido por la comunidad y está ampliamente disponible en plataformas como **Kaggle**, que es de donde se obtuvo para este proyecto.

### 1.2. Archivos Utilizados

Para la construcción del dataset base, se seleccionaron 8 de los archivos CSV disponibles, ya que contienen la información más relevante para la predicción a nivel de carrera-piloto:

| Archivo | Filas Aprox. | Descripción |
|---|---|---|
| `circuits.csv` | 77 | Información geográfica y técnica de los circuitos. |
| `constructors.csv` | 212 | Datos de los constructores/escuderías. |
| `drivers.csv` | 861 | Información demográfica de los pilotos. |
| `races.csv` | 1,125 | Catálogo histórico de carreras. |
| `results.csv` | 26,759 | Resultados detallados por piloto en cada carrera. **(Tabla Central)** |
| `qualifying.csv` | 10,494 | Tiempos y posiciones de las sesiones de clasificación. |
| `pit_stops.csv` | 11,371 | Registros detallados de las paradas en boxes. |
| `lap_times.csv` | 589,081 | Tiempos por vuelta de cada piloto en cada carrera. |

### 1.3. Estructura Relacional

El conjunto de datos conforma una base de datos relacional. La tabla `results.csv` actúa como el núcleo, conectando las demás entidades a través de claves foráneas.

```
circuits (1) --< (N) races (1) --< (N) results (N) >-- (1) drivers
                                        |     |
                                        |     '---- (1) constructors
                                        |
                                        '--< (N) qualifying
                                        '--< (N) pit_stops
                                        '--< (N) lap_times
```


## 2. Selección e Integración de Tablas

### 2.1. Criterio de Selección

Se seleccionaron los 8 archivos mencionados porque contienen variables directamente relacionadas con el rendimiento en una carrera específica. Se excluyeron tablas de clasificación de temporada (`driver_standings.csv`, `constructor_standings.csv`) para el modelo base, ya que el objetivo es predecir el resultado de una carrera individual, aunque estas variables se proponen como mejoras futuras (ver Sección 5.2).

### 2.2. Proceso de Unión (Joins)

La integración de las distintas fuentes de datos se lleva a cabo mediante una fusión secuencial de DataFrames en la biblioteca `pandas`. Se adopta una estrategia de unión mixta, combinando operaciones `inner join` y `left join` para maximizar tanto la integridad referencial como el volumen de datos disponibles para el análisis.

Inicialmente, se realizan uniones de tipo `inner` entre la tabla central `results` y las tablas `drivers`, `races`, `circuits` y `constructors`. Esta decisión metodológica garantiza que el conjunto de datos base contenga únicamente registros con una integridad referencial completa, es decir, cada resultado debe estar asociado a un piloto, carrera, circuito y constructor válidos. Posteriormente, se procede a fusionar las tablas de datos agregados (`lap_stats`, `pit_stats`, `quali_stats`) mediante operaciones `left join`. Este enfoque permite conservar todos los registros de la tabla fusionada principal, incluso si no existen datos correspondientes de vueltas, paradas o clasificación, evitando así la pérdida de información valiosa.

**Fragmento de Código de Fusión:**
```python
# Fusión central: results como tabla principal
data = results.merge(drivers, on='driverId')
data = data.merge(races, on='raceId')
data = data.merge(circuits, on='circuitId')
data = data.merge(constructors, on='constructorId')

# Fusiones con tablas agregadas (para no perder registros)
data = data.merge(lap_stats, on=['raceId', 'driverId'], how='left')
data = data.merge(pit_stats, on=['raceId', 'driverId'], how='left')
data = data.merge(quali_stats, on=['raceId', 'driverId'], how='left')
```

---

## 3. Limpieza de Datos

### 3.1. Tratamiento de Valores Faltantes

El primer paso en la fase de limpieza consiste en la normalización de los valores faltantes. Los archivos CSV originales utilizan la cadena de texto `\N`, una convención heredada de PostgreSQL, para representar datos nulos. Se implementó un tratamiento sistemático para reemplazar todas las instancias de `\N` por el marcador estándar `numpy.nan`. Esta conversión es un prerrequisito fundamental para que las funciones de las bibliotecas `pandas` y `scikit-learn`, tales como `dropna()`, `fillna()` y `StandardScaler`, puedan detectar y gestionar adecuadamente los datos ausentes durante las fases subsecuentes del preprocesamiento y modelado.
  ```python
  df.replace('\\N', np.nan, inplace=True)
  ```
- **Justificación**: Esta conversión es imperativa para que las funciones de `pandas` y `scikit-learn` (ej. `dropna()`, `fillna()`, `StandardScaler`) detecten y manejen correctamente los datos ausentes.

### 3.2. Eliminación de Registros Incompletos

- **Decisión**: Después de la fusión y la selección de características, se eliminan las filas que contienen al menos un valor nulo en las variables predictoras seleccionadas o en la variable objetivo.
  ```python
  data_clean = data[['win'] + vars_disponibles].dropna()
  ```
- **Justificación**: Para el modelo base, se opta por un enfoque de "casos completos" (`complete-case analysis`) para asegurar que el modelo se entrene con datos de alta calidad sin necesidad de imputación, que podría introducir sesgos. Esto reduce el tamaño del dataset pero aumenta la fiabilidad de las muestras restantes.

---

## 4. Definición de la Variable Objetivo

- **Construcción**: La variable objetivo `win` es binaria y se deriva de la columna `positionOrder` del archivo `results.csv`.
  ```python
  data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
  ```
- **Definición**:
  - `win = 1`: El piloto finalizó la carrera en primera posición.
  - `win = 0`: El piloto finalizó en cualquier otra posición.
- **Justificación**: Esta formulación transforma el problema de predecir una posición exacta (regresión o clasificación multiclase) en un problema de clasificación binaria, que responde directamente a la pregunta de investigación: ¿qué factores determinan una victoria?

---

## 5. Ingeniería de Características

### 5.1. Variables del Modelo Base

Para el primer modelo, se seleccionaron y crearon las siguientes variables:

| Variable | Descripción | Justificación Conceptual |
|---|---|---|
| `age` | Edad del piloto en años, calculada a partir de su fecha de nacimiento (`dob`). | Proxy de la experiencia y la condición física. La relación rendimiento-edad puede no ser lineal (pico de carrera). |
| `grid` | Posición de salida en la parrilla. | Una de las variables más predictivas. Una mejor posición de salida reduce la distancia a recorrer y los riesgos de incidentes. |
| `laps_completed` | Vueltas completadas en la carrera. | Indica fiabilidad y capacidad de terminar la carrera. |
| `lap_time_mean` | Tiempo medio por vuelta. | Métrica directa del ritmo de carrera del piloto. |
| `lap_time_std` | Desviación estándar de los tiempos de vuelta. | Mide la consistencia del piloto durante la carrera. |
| `lap_time_min` | Mejor tiempo de vuelta. | Representa el rendimiento máximo potencial del conjunto coche-piloto. |
| `pit_stops_count` | Número de paradas en boxes. | Proxy de la estrategia de carrera. Más paradas pueden indicar problemas o una estrategia agresiva. |
| `quali_position` | Posición final en la clasificación. | Similar a `grid`, pero puede contener información adicional de las sesiones Q1, Q2, Q3. |
| `nationality` | Nacionalidad del piloto (categórica). | Puede correlacionar con escuelas de pilotaje, patrocinadores o acceso a equipos de primer nivel. |
| `constructorId` | ID del equipo/escudería (categórica). | Proxy fundamental de la calidad del monoplaza (motor, chasis, aerodinámica). |

### 5.2. Propuesta de Variables Adicionales (Para Futuras Iteraciones)

El archivo `variables_adicionales_analisis.md` propone un conjunto de características de alto valor para mejorar el rendimiento del modelo. Su implementación es una recomendación prioritaria.

| Variable Propuesta | Descripción | Justificación Conceptual | Prioridad |
|---|---|---|---|
| **`experiencia_circuito`** | Nº de carreras previas del piloto en ese circuito. | El conocimiento del trazado, las líneas de carrera y las condiciones locales es una ventaja competitiva clave. | **Imperativa** |
| `experiencia_constructor` | Nº de carreras previas del piloto con ese equipo. | Mide la adaptación del piloto a la tecnología, los procesos y la cultura del equipo. | Alta |
| `ranking_constructor` | Posición del equipo en el mundial de constructores. | Proxy directo de la competitividad y los recursos del equipo en esa temporada. | Muy Alta |
| `ranking_piloto` | Posición del piloto en el mundial de pilotos. | Captura el estado de forma actual ("momentum"), la confianza y el estatus dentro del equipo. | Alta |
| `edad_inicio_carrera` | Edad exacta del piloto en el día de la carrera. | Más precisa que la edad actual, captura la evolución del rendimiento a lo largo del tiempo. | Alta |

**Implementación sugerida para la variable recomendada:**
```python
# Ordenar cronológicamente para asegurar que el cálculo acumulativo sea correcto
df = df.sort_values(['year', 'date'])

# Calcular la experiencia acumulada por piloto en cada circuito
df['experiencia_circuito'] = df.groupby(['driverId', 'circuitId']).cumcount()
```

---

## 6. Codificación y Transformación de Variables

### 6.1. Variables Categóricas

- **Técnica**: One-Hot Encoding, implementada con `pandas.get_dummies()`.
- **Variables Afectadas**: `nationality` y `constructorId`.
- **Descripción**: Esta técnica convierte cada categoría en una nueva columna binaria (0 o 1). Por ejemplo, `nationality` se expande a `nationality_British`, `nationality_German`, etc.
- **Justificación**: Los modelos de regresión y las redes neuronales requieren entradas numéricas. One-Hot Encoding es adecuado porque no introduce un orden artificial entre las categorías (a diferencia de Label Encoding).

### 6.2. Normalización/Estandarización

- **Método**: Estandarización, utilizando `StandardScaler` de `scikit-learn`.
- **Descripción**: Transforma cada variable numérica para que tenga una media de 0 y una desviación estándar de 1.
  ```python
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```
- **Justificación**: Es un requisito para modelos sensibles a la escala de las características, como la Regresión Lineal/Logística (especialmente con regularización) y las Redes Neuronales, ya que asegura una convergencia más rápida y estable del optimizador.

---

## 7. Análisis Exploratorio Mínimo (EDA)

### 7.1. Estadísticas Descriptivas

El script de preparación genera un dataset procesado (`f1_dataset_processed.csv`) y un dataset "crudo" (`f1_dataset_raw.csv`) tras la limpieza. El dataset limpio contiene **~8,000 filas** y **15 columnas** (1 `win` + 12 numéricas + 2 categóricas) antes del one-hot encoding.

### 7.2. Distribución de la Variable Objetivo

El análisis de la variable `win` revela un fuerte desbalance de clases, lo cual es inherente a la naturaleza del problema.

- **Observaciones Totales (Dataset Limpio)**: ~8,000
- **Victorias (Clase 1)**: ~400 (~5%)
- **No Victorias (Clase 0)**: ~7,600 (~95%)

**Comentario sobre Desbalance:**
Este desbalance es esperado, ya que en cada carrera solo hay un ganador y ~20 no ganadores. Implicaciones para el modelado:
1. La métrica de **Accuracy no es fiable**. Un modelo que siempre predice "No Gana" tendría un 95% de accuracy.
2. Se deben utilizar métricas como **F1-Score**, **AUC-ROC** y **Precision-Recall AUC**.
3. Es necesario aplicar técnicas de manejo de desbalance, como el parámetro `class_weight='balanced'` en los modelos.

---

## 8. Partición del Conjunto de Datos

- **Método**: Se utiliza la función `train_test_split` de `scikit-learn`.
- **Proporciones**:
  - **Conjunto de Entrenamiento**: 80% de los datos.
  - **Conjunto de Prueba**: 20% de los datos.
- **Criterio de Partición**:
  - `random_state=42`: Se fija una semilla para garantizar la reproducibilidad de la partición.
  - `stratify=y`: Se realiza una partición estratificada según la variable objetivo `y`. Esto asegura que la proporción de victorias y no victorias sea la misma en el conjunto de entrenamiento y en el de prueba, lo cual es crucial en datasets desbalanceados.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 9. Producto Final

- **Estructura**: El producto final de este pipeline es un DataFrame de `pandas` (guardado como `f1_dataset_processed.csv`) con aproximadamente **8,000 observaciones** y **más de 200 variables** (tras el one-hot encoding).
- **Contenido**: Cada fila representa una participación de un piloto en una carrera y contiene las variables predictoras numéricas estandarizadas y las variables categóricas codificadas.
- **Confirmación**: Este dataset está limpio, preprocesado y estructurado, listo para ser utilizado como entrada para entrenar el primer modelo de Regresión Logística y los modelos subsecuentes.

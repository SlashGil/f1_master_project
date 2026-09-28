# Documentación del Dataset: dataset_tesis_f1.csv

## 1. Descripción General

Este dataset constituye la base de datos unificada para el desarrollo de modelos 
predictivos de victorias en carreras de Fórmula 1. Se ha generado a partir de 
la integración y transformación de 8 archivos CSV originales mediante un pipeline 
estándarizado de procesamiento de datos.

### 1.1 Identificación del Dataset
- **Nombre del archivo**: `dataset_tesis_f1.csv`
- **Versión**: 1.0
- **Fecha de generación**: 2026-03-04 21:10:54
- **Propósito**: Base de datos compartida para entrenamiento de tres modelos ML

### 1.2 Dimensiones del Dataset
- **Observaciones**: 655,189
- **Variables**: 56 (incluyendo target)
- **Variables predictoras**: 55
- **Variable objetivo**: 1 (`win`)

---

## 2. Variables del Dataset

### 2.1 Variable Objetivo

| Variable | Descripción | Tipo | Valores | Balance |
|----------|-------------|------|---------|---------|
| `win` | El piloto ganó la carrera | Binaria | 0=No, 1=Sí | 5.29% positivos |

### 2.2 Variables Predictoras Numéricas

| Variable | Descripción | Fuente | Unidad |
|----------|-------------|--------|--------|
| age | Edad actual del piloto | drivers.csv | años |
| grid | Posición de inicio | results.csv | posición |
| fastestLapSpeed | Velocidad en vuelta más rápida | results.csv | km/h |
| milliseconds_pit_stop | Duración del primer pit stop | pit_stops.csv | ms |
| milliseconds_lap_time | Tiempo de una vuelta | lap_times.csv | ms |

### 2.3 Variables Predictoras Categóricas (Transformadas a One-Hot)

| Variable Original | Descripción | Cardinalidad | Transformación |
|------------------|-------------|--------------|----------------|
| nationality | Nacionalidad del piloto | 27 países | One-hot (nationality_COUNTRY) |
| constructorId | Constructor/escudería | 23 escuderías | One-hot (constructorId_N) |

---

## 3. Proceso de Generación del Dataset

### 3.1 Archivos Fuente

Se utilizaron 8 archivos CSV originales:

1. **circuits.csv** - Información de circuitos (77 filas)
2. **results.csv** - Resultados de carreras (26759 filas)
3. **drivers.csv** - Información de pilotos (861 filas)
4. **pit_stops.csv** - Paradas en boxes (11371 filas)
5. **races.csv** - Catálogo de carreras (1125 filas)
6. **qualifying.csv** - Datos de clasificación (10494 filas)
7. **lap_times.csv** - Tiempos de vuelta (589081 filas)
8. **constructors.csv** - Datos de constructores (212 filas)

### 3.2 Pipeline de Preprocesamiento

#### Paso 1: Limpieza de Valores Nulos
- Sustitución de '\N' (formato PostgreSQL) por `NaN` (valor nulo pandas)
- Aplicado a: results, drivers, pit_stops, races, qualifying, lap_times

#### Paso 2: Ingeniería de Características
- Cálculo de edad actual desde fecha de nacimiento: `(fecha_actual - dob) / 365`
- Renombrado de columnas ambiguas:
  - pit_stops: `milliseconds` → `milliseconds_pit_stop`
  - lap_times: `milliseconds` → `milliseconds_lap_time`

#### Paso 3: Fusión de Datasets (6 JOINs)

| Paso | Operación | Tipo | Claves | Resultado |
|------|-----------|------|--------|-----------|
| 1 | results ← drivers | inner | driverId | 26759 → 655189 + (N/A) |
| 2 | ← races | inner | raceId | Preserva integridad |
| 3 | ← circuits | inner | circuitId | Preserva integridad |
| 4 | ← pit_stops | left | raceId, driverId | Introduce multiplicidad |
| 5 | ← qualifying | left | raceId, driverId | 0-1 relación |
| 6 | ← lap_times | left | raceId, driverId | Introduce multiplicidad alta |

**Justificación de estrategia de fusión:**
- **Inner joins (pasos 1-3)**: Garantizan que las observaciones cumplan criterios mínimos 
  de integridad (existen tanto piloto como carrera y circuito)
- **Left joins (pasos 4-6)**: Preservan observaciones aunque falten detalles de carrera
  (pit_stops, qualifying, lap_times), permitiendo modelos aprender con información
  parcial. Estos joins introducen multiplicidad N:M (múltiples filas por campaña/piloto 
  debido a múltiples vueltas o paradas)

#### Paso 4: Limpieza Final
- Eliminación de observaciones con valores faltantes en features o target
- Filas eliminadas: 289,091 (30.61%)
- Filas retenidas: 655,189 (69.39%)

#### Paso 5: Codificación One-Hot
- Transformación de variables categóricas a dummy variables
- `nationality` → 27 columnas binarias
- `constructorId` → 23 columnas binarias
- Resultado: 56 columnas totales

---

## 4. Estadísticas del Dataset

### 4.1 Distribución de la Variable Objetivo

| Métrica | Valor |
|---------|-------|
| Total observaciones | 655,189 |
| Victorias (positivos) | 49,984 (5.29%) |
| Sin victorias (negativos) | 894,296 (94.71%) |
| Ratio desbalance | 1:17.9 : (negativos por cada positivo) |

### 4.2 Composición del Dataset

| Tipo de Feature | Categoría | Cantidad |
|-----------------|-----------|----------|
| Numéricas | Variables continuas/discretas | 5 |
| Categóricas | nationality (one-hot) | 27 |
| Categóricas | constructorId (one-hot) | 23 |
| **Total Features** | - | 55 |

---

## 5. Uso del Dataset

### 5.1 Modelos Objetivos

Este dataset está diseñado para entrenar tres modelos de aprendizaje automático:

1. **Regresión Lineal** - Modelo interpretable con coeficientes que indican la 
   influencia directa de cada variable en la probabilidad de victoria

2. **Random Forest** - Modelo de ensamble robusto que captura relaciones no
   lineales e interacciones entre variables mediante árboles de decisión múltiples

3. **Perceptrón Multicapa (MLP)** - Red neuronal profunda con capa(s) oculta(s)
   que aprende representaciones complejas de los datos

### 5.2 Carga del Dataset en Python

```python
import pandas as pd

# Cargar el dataset
df = pd.read_csv('dataset_tesis_f1.csv')

# Separar features y target
X = df.drop('win', axis=1)
y = df['win']

# El dataset ya está preprocesado (one-hot encoding aplicado)
# Solo requiere escalado (StandardScaler) para algunos modelos
```

### 5.3 Observaciones Importantes

⚠️ **Advertencia - Fuga Temporal Potencial**:

El dataset contiene variables que solo se conocen **después** de la carrera:
- `fastestLapSpeed`: Velocidad máxima registrada durante la carrera
- `milliseconds_lap_time`: Tiempos específicos de vueltas
- `milliseconds_pit_stop`: Duraciones de paradas en boxes

Si el objetivo es **predicción a priori** (antes de la carrera), estas variables
no deben utilizarse. Para un modelo de predicción a priori, se recomienda usar solo:
- `age` (edad del piloto)
- `constructorId` (escudería)
- `nationality` (nacionalidad)
- `grid` (posición de clasificación)

Si el objetivo es **análisis a posteriori** (interpretación de factores que determinaron
la victoria), el dataset actual es apropiado.

⚠️ **Datos Desbalanceados**:

La variable objetivo `win` presenta un fuerte desbalance (~95% negativos, 5% positivos).
Se recomienda:
- Usar métricas apropiadas: F1-score, AUC-ROC, precision-recall
- Considerar técnicas de balanceo: class_weight, oversampling (SMOTE), undersampling
- Interpretar con cautela la accuracy (puede ser engañosa)

---

## 6. Archivos Generados

El proceso de preparación de datos genera los siguientes archivos:

| Archivo | Descripción | Formato |
|---------|-------------|---------|
| `dataset_tesis_f1.csv` | Dataset principal listo para modelado | CSV |
| `metadata_dataset.csv` | Metadatos completos del dataset | CSV |
| `estadisticas_dataset.csv` | Estadísticas descriptivas | CSV |
| `DOCUMENTACION_DATASET_TESIS.md` | Este documento | Markdown |

---

## 7. Referencias

- Fuente de datos: F1 World Championship Historical Database
- Bibliotecas utilizadas: pandas, NumPy
- Metodología: Standard preprocessing pipeline for tabular data

---

**Generado automáticamente por: preparar_datos_tesis.py**  
**Fecha**: 2026-03-04 21:10:54  
**Versión del documento**: 1.0

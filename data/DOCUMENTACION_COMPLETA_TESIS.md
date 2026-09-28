# Documentación Completa de Preparación de Dataset para Tesis
## Procesamiento de Datos de Fórmula 1 para Predicción **Pre-Carrera**

---

## Tabla de Contenidos

1. [Introducción y Contexto](#1-introducción-y-contexto)
2. [Cambios Metodológicos Clave](#2-cambios-metodológicos-clave)
3. [Fuentes de Datos Utilizadas](#3-fuentes-de-datos-utilizadas)
4. [Pipeline de Preparación de Datos](#4-pipeline-de-preparación-de-datos)
5. [Variables Finales del Dataset](#5-variables-finales-del-dataset)
6. [Estrategia de Split Temporal y Escalado](#6-estrategia-de-split-temporal-y-escalado)
7. [Estadísticas del Dataset Final](#7-estadísticas-del-dataset-final)
8. [Implicaciones para Modelado](#8-implicaciones-para-modelado)
9. [Limitaciones y Trabajo Futuro](#9-limitaciones-y-trabajo-futuro)
10. [Conclusión](#10-conclusión)

---

## 1. Introducción y Contexto

Esta versión de la documentación actualiza la metodología del proyecto para alinearla con una evaluación más realista de predicción de victorias en Fórmula 1.

El enfoque ahora es explícitamente **pre-carrera**, por lo que:

- Se eliminan variables que solo existen durante o después de la carrera.
- Se evita el split aleatorio 80/20 para no mezclar información futura con pasado.
- Se adopta división temporal para entrenamiento y prueba.

---

## 2. Cambios Metodológicos Clave

### 2.1 Cambio de split aleatorio a split temporal

**Antes (no recomendado para este caso):**
- `train_test_split(..., test_size=0.2, random_state=42)`

**Ahora (recomendado):**
- Orden cronológico por `year`, `round`, `raceId`
- Corte temporal 80/20 (pasado para train, futuro para test)

Esto evita contaminación temporal y mejora la validez de evaluación.

### 2.2 Eliminación de fuga temporal de variables

Se excluyen del dataset final las variables post-carrera:

- `fastestLapSpeed`
- `milliseconds_pit_stop`
- `milliseconds_lap_time`
- `positionOrder` como predictor (solo se usa para derivar `win`)

### 2.3 Escalado correcto

`StandardScaler` debe ejecutarse **después** del split temporal:

1. `fit` solo con entrenamiento.
2. `transform` en entrenamiento y prueba.

---

## 3. Fuentes de Datos Utilizadas

Para esta versión pre-carrera se usan las tablas mínimas necesarias:

1. `results.csv`
2. `drivers.csv`
3. `races.csv`

Motivo: son suficientes para construir features pre-carrera y target sin incluir señales post-hoc.

---

## 4. Pipeline de Preparación de Datos

### Paso 1: Carga y limpieza

- Carga de `results`, `drivers`, `races`
- Reemplazo de `\N` por `NaN`
- Conversión de tipos (`positionOrder`, `grid`, `constructorId`, `year`, `round`, `date`, `dob`)

### Paso 2: Fusión

- Join `results` + `drivers` por `driverId`
- Join resultado + `races` por `raceId`

### Paso 3: Ingeniería de features pre-carrera

- `age`: edad del piloto al momento de la carrera
- `grid`: posición de salida
- `year`, `round`: contexto temporal
- `nationality`, `constructorId`: variables categóricas (luego one-hot)

### Paso 4: Variable objetivo

- `win = 1` si `positionOrder == 1`, en otro caso `0`

### Paso 5: Limpieza final y encoding

- Eliminación de filas con NA en variables base
- Filtro `grid > 0`
- One-hot encoding para `nationality` y `constructorId`

### Paso 6: Orden temporal y guardado

- Orden por `year`, `round`, `raceId`
- Guardado de:
  - `dataset_tesis_f1.csv`
  - `metadata_dataset.csv`
  - `estadisticas_dataset.csv`
  - `DOCUMENTACION_DATASET_TESIS.md`

---

## 5. Variables Finales del Dataset

### 5.1 Numéricas base

- `age`
- `grid`
- `year`
- `round`

### 5.2 Categóricas (one-hot)

- `nationality_*`
- `constructorId_*`

### 5.3 Variable objetivo

- `win`

### 5.4 Campo de orden temporal

- `raceId` (se conserva para orden y split temporal)

---

## 6. Estrategia de Split Temporal y Escalado

### 6.1 Regla de split

1. Ordenar por `year`, `round`, `raceId`.
2. Tomar 80% inicial como entrenamiento.
3. Tomar 20% final como prueba.

### 6.2 Escalado

Ejecutar solo tras la división:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

No hacer `fit` del scaler con datos de test.

---

## 7. Estadísticas del Dataset Final

Generación actual:

- **Archivo**: `dataset_tesis_f1.csv`
- **Observaciones**: 25,121
- **Columnas totales**: 247
- **Features (sin target)**: 246
- **Victorias (`win=1`)**: 1,128 (4.49%)

El desbalance de clases se mantiene (propio del dominio de F1).

---

## 8. Implicaciones para Modelado

- La evaluación temporal representa mejor un escenario real de predicción futura.
- El dataset es apto para comparar modelos sin fuga temporal explícita.
- Métricas recomendadas:
  - F1-score (clase positiva)
  - AUC-ROC
  - AUC-PR
  - Precision/Recall

---

## 9. Limitaciones y Trabajo Futuro

1. **Desbalance de clases** (~95/5): considerar `class_weight`, umbral adaptativo, calibración.
2. **Señales históricas agregadas**: incorporar historial acumulado de piloto/constructor usando solo pasado.
3. **Validación temporal avanzada**: usar `TimeSeriesSplit` con múltiples folds para robustez.

---

## 10. Conclusión

El proyecto queda actualizado a una metodología coherente con predicción **antes de la carrera**:

- Se mantiene el nombre final del dataset: `dataset_tesis_f1.csv`.
- Se elimina fuga temporal por variables post-carrera.
- Se migra de split aleatorio a split temporal.
- Se documenta y aplica el escalado correctamente después del split.

---

**Generado/actualizado para tesis**  
**Fecha:** 2026-04-16  
**Versión del documento:** 3.0

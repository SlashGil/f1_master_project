# Documentación del Dataset: dataset_tesis_f1.csv

## 1. Objetivo de esta versión

Esta versión del dataset está diseñada para **predicción pre-carrera** y corrige dos puntos metodológicos clave:

1. **No usar split aleatorio 80/20** para evaluación principal.
2. **Eliminar variables con fuga temporal** (información disponible durante o después de la carrera).

## 2. Principios metodológicos aplicados

- **Nombre final preservado**: `dataset_tesis_f1.csv`
- **Split recomendado**: **TimeSeriesSplit** o corte temporal ordenado por `year`, `round`, `raceId`
- **Escalado correcto**: `StandardScaler` se ajusta **solo en train**, y luego transforma test

## 3. Features incluidas (pre-carrera)

- Numéricas base: `age`, `grid`, `year`, `round`
- Categóricas (one-hot): `nationality`, `constructorId`
- Target: `win`

## 4. Features excluidas por fuga temporal

- `fastestLapSpeed`
- `milliseconds_pit_stop`
- `milliseconds_lap_time`
- `positionOrder` como predictor (se usa solo para derivar `win`)

## 5. Tamaño del dataset generado

- Observaciones: 25,121
- Columnas totales: 247
- Features (sin target): 246
- Victorias: 1,128 (4.49%)

## 6. Ejemplo correcto de split temporal + escalado

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar dataset ya preprocesado
df = pd.read_csv('dataset_tesis_f1.csv')

# Orden temporal
df = df.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)

# Split temporal 80/20 (NO random)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns=['win'])
y_train = train_df['win']
X_test = test_df.drop(columns=['win'])
y_test = test_df['win']

# Escalado correcto: fit solo con train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

**Generado automáticamente por:** `preparar_datos_tesis.py`  
**Fecha:** 2026-04-16 11:25:10  
**Versión del documento:** 2.0

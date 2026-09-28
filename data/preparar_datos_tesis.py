"""
Script de Preparación de Datos para Tesis F1 (versión pre-carrera)
==================================================================

Objetivo de esta versión:
- Mantener el nombre final del dataset: dataset_tesis_f1.csv
- Eliminar variables con fuga temporal (información de carrera o post-carrera)
- Preparar datos para evaluación temporal (Time Series Split)
- Documentar explícitamente que el escalado se hace DESPUÉS del split temporal
"""

from __future__ import annotations

from datetime import datetime
import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "version_dataset": "2.0",
    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "descripcion": "Dataset pre-carrera para predicción de victoria en F1 (sin fuga temporal)",
    "metodo_split_recomendado": "TimeSeriesSplit / corte temporal por año-ronda (no random 80/20)",
    "escalado_recomendado": "Ajustar scaler SOLO con train y luego transformar test",
}

NOMBRE_DATASET = "dataset_tesis_f1.csv"
NOMBRE_METADATA = "metadata_dataset.csv"
NOMBRE_ESTADISTICAS = "estadisticas_dataset.csv"
NOMBRE_DOC = "DOCUMENTACION_DATASET_TESIS.md"


# ============================================================================
# CARGA
# ============================================================================

print("=" * 90)
print(" PREPARACIÓN DE DATOS F1 - VERSIÓN PRE-CARRERA (SIN FUGA TEMPORAL)")
print("=" * 90)

print("\n[1/7] Cargando archivos fuente...")
results = pd.read_csv("results.csv")
drivers = pd.read_csv("drivers.csv")
races = pd.read_csv("races.csv")

print(f"  ✓ results.csv : {results.shape[0]:,} x {results.shape[1]}")
print(f"  ✓ drivers.csv : {drivers.shape[0]:,} x {drivers.shape[1]}")
print(f"  ✓ races.csv   : {races.shape[0]:,} x {races.shape[1]}")


# ============================================================================
# LIMPIEZA
# ============================================================================

print("\n[2/7] Limpieza y normalización...")
for name, df in [("results", results), ("drivers", drivers), ("races", races)]:
    df.replace("\\N", np.nan, inplace=True)
    print(f"  ✓ {name:<8} normalizado (\\N -> NaN)")

results["positionOrder"] = pd.to_numeric(results["positionOrder"], errors="coerce")
results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
results["constructorId"] = pd.to_numeric(results["constructorId"], errors="coerce")

races["year"] = pd.to_numeric(races["year"], errors="coerce")
races["round"] = pd.to_numeric(races["round"], errors="coerce")
races["date"] = pd.to_datetime(races["date"], errors="coerce")

drivers["dob"] = pd.to_datetime(drivers["dob"], errors="coerce")


# ============================================================================
# FUSIÓN PRE-CARRERA
# ============================================================================

print("\n[3/7] Integrando datos mínimos necesarios (pre-carrera)...")
data = results.merge(
    drivers[["driverId", "nationality", "dob"]],
    on="driverId",
    how="inner",
).merge(
    races[["raceId", "year", "round", "date"]],
    on="raceId",
    how="inner",
)

print(f"  ✓ Filas tras fusión: {len(data):,}")


# ============================================================================
# FEATURES PRE-CARRERA + TARGET
# ============================================================================

print("\n[4/7] Creando variables pre-carrera y target...")

# Edad del piloto AL MOMENTO de la carrera (más correcto que edad actual)
data["age"] = ((data["date"] - data["dob"]).dt.days / 365.25).astype(float)

# Target binario
# 1 = ganó la carrera, 0 = no ganó
data["win"] = (data["positionOrder"] == 1).astype(int)

# Variables permitidas para predicción pre-carrera
# (eliminamos explícitamente variables post-carrera como fastestLapSpeed, lap_times, pit_stops, etc.)
features_numericas = ["age", "grid", "year", "round"]
features_categoricas = ["nationality", "constructorId"]

columnas_base = [
    "raceId",
    *features_numericas,
    *features_categoricas,
    "win",
]

antes_dropna = len(data)
data = data[columnas_base].dropna().copy()

# Filtro de calidad mínima: posiciones de salida válidas (>0)
data = data[data["grid"] > 0].copy()

despues_dropna = len(data)
print(f"  ✓ Filas antes de limpieza final: {antes_dropna:,}")
print(f"  ✓ Filas después de limpieza: {despues_dropna:,}")


# ============================================================================
# ONE-HOT ENCODING
# ============================================================================

print("\n[5/7] Aplicando one-hot encoding...")

X = data[["raceId", *features_numericas, *features_categoricas]].copy()
y = data["win"].copy()

X_encoded = pd.get_dummies(X, columns=features_categoricas, dtype=float)

# Orden temporal canónico para split por tiempo
sort_cols = ["year", "round", "raceId"] if {"year", "round", "raceId"}.issubset(X_encoded.columns) else ["raceId"]

dataset_final = X_encoded.copy()
dataset_final["win"] = y.values
dataset_final = dataset_final.sort_values(sort_cols).reset_index(drop=True)


# ============================================================================
# GUARDADO
# ============================================================================

print("\n[6/7] Guardando archivos...")
dataset_final.to_csv(NOMBRE_DATASET, index=False)

n_victorias = int(dataset_final["win"].sum())
total = len(dataset_final)
pct_victorias = (n_victorias / total * 100) if total else 0.0
n_no_victorias = total - n_victorias

# Metadata
metadata_rows = [
    {"tipo": "configuracion", "clave": "version", "valor": CONFIG["version_dataset"]},
    {"tipo": "configuracion", "clave": "fecha_generacion", "valor": CONFIG["fecha"]},
    {"tipo": "configuracion", "clave": "descripcion", "valor": CONFIG["descripcion"]},
    {
        "tipo": "metodologia",
        "clave": "split_recomendado",
        "valor": CONFIG["metodo_split_recomendado"],
    },
    {
        "tipo": "metodologia",
        "clave": "escalado_recomendado",
        "valor": CONFIG["escalado_recomendado"],
    },
    {
        "tipo": "metodologia",
        "clave": "features_excluidas_por_fuga",
        "valor": "fastestLapSpeed, milliseconds_pit_stop, milliseconds_lap_time, positionOrder (como feature)",
    },
    {"tipo": "dataset_final", "clave": "nombre_archivo", "valor": NOMBRE_DATASET},
    {"tipo": "dataset_final", "clave": "observaciones", "valor": total},
    {"tipo": "dataset_final", "clave": "columnas_totales", "valor": len(dataset_final.columns)},
    {
        "tipo": "dataset_final",
        "clave": "features_numericas_base",
        "valor": ", ".join(features_numericas),
    },
    {
        "tipo": "dataset_final",
        "clave": "features_categoricas_base",
        "valor": ", ".join(features_categoricas),
    },
    {"tipo": "estadisticas_clases", "clave": "total_observaciones", "valor": total},
    {"tipo": "estadisticas_clases", "clave": "victorias", "valor": n_victorias},
    {"tipo": "estadisticas_clases", "clave": "sin_victorias", "valor": n_no_victorias},
    {"tipo": "estadisticas_clases", "clave": "porcentaje_victorias", "valor": f"{pct_victorias:.2f}%"},
]

pd.DataFrame(metadata_rows).to_csv(NOMBRE_METADATA, index=False)

# Estadísticas resumidas
estadisticas_df = pd.DataFrame(
    {
        "metrica": [
            "Total observaciones",
            "Total victorias",
            "Total sin victorias",
            "Porcentaje victorias",
            "Porcentaje sin victorias",
            "Features totales",
            "Split recomendado",
            "Escalado correcto",
        ],
        "valor": [
            total,
            n_victorias,
            n_no_victorias,
            f"{pct_victorias:.2f}%",
            f"{(100 - pct_victorias):.2f}%",
            len(dataset_final.columns) - 1,
            "TimeSeriesSplit / corte temporal",
            "Fit scaler en train, transform en test",
        ],
        "comentario": [
            "Número total de observaciones en el dataset",
            "Casos positivos (piloto ganó)",
            "Casos negativos (piloto no ganó)",
            "Porcentaje de casos positivos",
            "Porcentaje de casos negativos",
            "Variables predictoras (excluyendo target)",
            "Evita mezclar futuro con pasado",
            "Evita fuga de información por preprocesamiento",
        ],
    }
)
estadisticas_df.to_csv(NOMBRE_ESTADISTICAS, index=False)

# Documentación técnica breve
doc = f"""# Documentación del Dataset: {NOMBRE_DATASET}

## 1. Objetivo de esta versión

Esta versión del dataset está diseñada para **predicción pre-carrera** y corrige dos puntos metodológicos clave:

1. **No usar split aleatorio 80/20** para evaluación principal.
2. **Eliminar variables con fuga temporal** (información disponible durante o después de la carrera).

## 2. Principios metodológicos aplicados

- **Nombre final preservado**: `{NOMBRE_DATASET}`
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

- Observaciones: {total:,}
- Columnas totales: {len(dataset_final.columns)}
- Features (sin target): {len(dataset_final.columns) - 1}
- Victorias: {n_victorias:,} ({pct_victorias:.2f}%)

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
**Fecha:** {CONFIG['fecha']}  
**Versión del documento:** 2.0
"""

with open(NOMBRE_DOC, "w", encoding="utf-8") as f:
    f.write(doc)

print(f"  ✓ {NOMBRE_DATASET}")
print(f"  ✓ {NOMBRE_METADATA}")
print(f"  ✓ {NOMBRE_ESTADISTICAS}")
print(f"  ✓ {NOMBRE_DOC}")


# ============================================================================
# RESUMEN
# ============================================================================

print("\n[7/7] Resumen final")
print("-" * 90)
print(f"✓ Dataset final: {NOMBRE_DATASET}")
print(f"  - Filas: {total:,}")
print(f"  - Columnas: {len(dataset_final.columns)}")
print(f"  - Victorias: {n_victorias:,} ({pct_victorias:.2f}%)")
print("✓ Metodología actualizada: split temporal + sin fuga de variables post-carrera")
print("✓ Escalado documentado correctamente: después del split temporal")
print("\nProceso completado.")

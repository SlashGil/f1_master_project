"""
Script de Preparación de Datos para Tesis F1
==============================================

Este script realiza la limpieza, transformación y preparación integral de los
datos de Fórmula 1 para el análisis predictivo de victorias.

OBJETIVO:
  Generar un dataset unificado (dataset_tesis_f1.csv) que servira como base
  para entrenar los tres modelos predictivos:
  1. Regresión Lineal
  2. Random Forest  
  3. Perceptrón Multicapa

OUTPUT PRINCIPAL:
  - dataset_tesis_f1.csv: Dataset limpio y consolidado con todas las features
  - metadata_dataset.csv: Documentación de variables y transformaciones

AUTOR: [Tú - para tu tesis]
FECHA: [Fecha actual]
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

print("="*90)
print(" PREPARACIÓN DE DATOS PARA TESIS F1 - GENERACIÓN DE DATASET BASE")
print("="*90)

# ============================================================================
# SECCIÓN 1: CONFIGURACIÓN Y METADATOS
# ============================================================================

print("\n[CONFIGURACIÓN]")
print("-" * 90)



# Configuración del proceso
CONFIG = {
    'version_dataset': '1.0',
    'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'descripcion': 'Dataset unificado para modelos predictivos de victorias en F1',
    'objetivo': 'Proporcionar un dataset limpio, estructurado y documentado para entrenamiento de modelos ML'
}

# Definición de las características que se incluirán en el dataset final
FEATURES_DEFINICION = {
    # Variables numéricas principales
    'age': {
        'descripcion': 'Edad actual del piloto en años',
        'fuente': 'drivers.csv (dob)',
        'transformacion': '(fecha_actual - fecha_nacimiento) / 365',
        'tipo': 'numérica continua',
        'rango': 'aprox. 20-45',
        'unidad': 'años'
    },
    'grid': {
        'descripcion': 'Posición de inicio en la carrera (de clasificación)',
        'fuente': 'results.csv',
        'transformacion': 'N/A (directo)',
        'tipo': 'numérica discreta',
        'rango': '1-20+',
        'unidad': 'posición'
    },
    'positionOrder': {
        'descripcion': 'Posición final en la carrera',
        'fuente': 'results.csv',
        'transformacion': 'N/A (directo)',
        'tipo': 'numérica discreta',
        'rango': '1-20+',
        'unidad': 'posición'
    },
    'fastestLapSpeed': {
        'descripcion': 'Velocidad en la vuelta más rápida de la carrera',
        'fuente': 'results.csv',
        'transformacion': 'N/A (directo)',
        'tipo': 'numérica continua',
        'rango': '~200-350',
        'unidad': 'km/h'
    },
    'milliseconds_pit_stop': {
        'descripcion': 'Duración del primer pit stop del piloto',
        'fuente': 'pit_stops.csv',
        'transformacion': 'Renombrado de "milliseconds"',
        'tipo': 'numérica continua',
        'rango': '~15000-40000',
        'unidad': 'milisegundos'
    },
    'milliseconds_lap_time': {
        'descripcion': 'Tiempo de una vuelta del piloto',
        'fuente': 'lap_times.csv',
        'transformacion': 'Renombrado de "milliseconds"',
        'tipo': 'numérica continua',
        'rango': '~60000-120000',
        'unidad': 'milisegundos'
    },
    
    # Variables categóricas (se convertirán a one-hot)
    'nationality': {
        'descripcion': 'Nacionalidad del piloto',
        'fuente': 'drivers.csv',
        'transformacion': 'One-hot encoding genera columnas nationality_COUNTRY',
        'tipo': 'categórica nominal',
        'cardinalidad': 'aprox. 50 países diferentes'
    },
    'constructorId': {
        'descripcion': 'Identificador del constructor/escudería',
        'fuente': 'results.csv y constructors.csv',
        'transformacion': 'One-hot encoding genera columnas constructorId_N',
        'tipo': 'categórica nominal',
        'cardinalidad': 'aprox. 220 constructores diferentes'
    }
}

# Variable objetivo
TARGET_DEFINICION = {
    'win': {
        'descripcion': 'Variable binaria: 1=el piloto ganó la carrera, 0=no ganó',
        'fuente': 'Derivado de results.csv(positionOrder)',
        'transformacion': '1 si positionOrder == 1, de lo contrario 0',
        'tipo': 'binaria',
        'valores': '0 (no ganó), 1 (ganó)',
        'balance': 'Aprox. 5% victorias, 95% sin victorias (clase desbalanceada)'
    }
}

print("Versión del dataset:", CONFIG['version_dataset'])
print("Fecha de generación:", CONFIG['fecha'])
print("\nFeatures seleccionadas:", len(FEATURES_DEFINICION))
print("Variable objetivo:", list(TARGET_DEFINICION.keys())[0])

# ============================================================================
# SECCIÓN 2: CARGA DE DATOS ORIGINALES
# ============================================================================

print("\n" + "="*90)
print("[2/7] CARGANDO DATASETS ORIGINALES")
print("="*90)

datasets_info = []

# Cargar todos los archivos CSV
print("\nCargando archivos...")
circuits = pd.read_csv('circuits.csv')
print(f"  ✓ circuits.csv           {circuits.shape[0]:>7} filas x {circuits.shape[1]} col")

results = pd.read_csv('results.csv')
print(f"  ✓ results.csv           {results.shape[0]:>7} filas x {results.shape[1]} col")

drivers = pd.read_csv('drivers.csv')
print(f"  ✓ drivers.csv           {drivers.shape[0]:>7} filas x {drivers.shape[1]} col")

pit_stops = pd.read_csv('pit_stops.csv')
print(f"  ✓ pit_stops.csv         {pit_stops.shape[0]:>7} filas x {pit_stops.shape[1]} col")

races = pd.read_csv('races.csv')
print(f"  ✓ races.csv             {races.shape[0]:>7} filas x {races.shape[1]} col")

qualifying = pd.read_csv('qualifying.csv')
print(f"  ✓ qualifying.csv       {qualifying.shape[0]:>7} filas x {qualifying.shape[1]} col")

lap_times = pd.read_csv('lap_times.csv')
print(f"  ✓ lap_times.csv         {lap_times.shape[0]:>7} filas x {lap_times.shape[1]} col")

constructors = pd.read_csv('constructors.csv')
print(f"  ✓ constructors.csv      {constructors.shape[0]:>7} filas x {constructors.shape[1]} col")

datasets_info = [
    {'archivo': 'circuits.csv', 'filas': len(circuits), 'columnas': len(circuits.columns)},
    {'archivo': 'results.csv', 'filas': len(results), 'columnas': len(results.columns)},
    {'archivo': 'drivers.csv', 'filas': len(drivers), 'columnas': len(drivers.columns)},
    {'archivo': 'pit_stops.csv', 'filas': len(pit_stops), 'columnas': len(pit_stops.columns)},
    {'archivo': 'races.csv', 'filas': len(races), 'columnas': len(races.columns)},
    {'archivo': 'qualifying.csv', 'filas': len(qualifying), 'columnas': len(qualifying.columns)},
    {'archivo': 'lap_times.csv', 'filas': len(lap_times), 'columnas': len(lap_times.columns)},
    {'archivo': 'constructors.csv', 'filas': len(constructors), 'columnas': len(constructors.columns)}
]

# ============================================================================
# SECCIÓN 3: LIMPIEZA Y PREPROCESAMIENTO
# ============================================================================

print("\n" + "="*90)
print("[3/7] LIMPIEZA Y PREPROCESAMIENTO DE DATOS")
print("="*90)

print("\n[3.1] Sustituyendo valores '\\N' por NaN (valor nulo de pandas)...")
# Reemplazar '\N' por NaN en todos los datasets
for df_name, df in [
    ('results', results),
    ('drivers', drivers),
    ('pit_stops', pit_stops),
    ('races', races),
    ('qualifying', qualifying),
    ('lap_times', lap_times)
]:
    df.replace('\\N', np.nan, inplace=True)
    nulos_anteriores = (df == '\\N').sum().sum()
    print(f"  ✓ {df_name:<12} reemplazados (total nulos: {df.isna().sum().sum()})")

print("\n[3.2] Calculando edad actual de pilotos...")
# Cálculo de edad actual
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
drivers['age'] = drivers['dob'].apply(lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else np.nan)
edad_stats = drivers['age'].describe()
print(f"  ✓ Edad creada. Rango: {edad_stats['min']:.0f} - {edad_stats['max']:.0f} años")
print(f"    Media: {edad_stats['mean']:.1f} ± {edad_stats['std']:.1f} años")

print("\n[3.3] Renombrando columnas ambiguas para evitar conflictos...")
# Renombrar columnas con nombres idénticos entre datasets
col_pit_anterior = 'milliseconds' in pit_stops.columns
col_lap_anterior = 'milliseconds' in lap_times.columns
pit_stops.rename(columns={'milliseconds': 'milliseconds_pit_stop'}, inplace=True)
lap_times.rename(columns={'milliseconds': 'milliseconds_lap_time'}, inplace=True)
print(f"  ✓ pit_stops:  'milliseconds' → 'milliseconds_pit_stop'")
print(f"  ✓ lap_times:  'milliseconds' → 'milliseconds_lap_time'")

# ============================================================================
# SECCIÓN 4: FUSIÓN DE DATASETS (JOIN OPERATIONS)
# ============================================================================

print("\n" + "="*90)
print("[4/7] FUSIÓN DE DATASETS")
print("="*90)

print("\nEstrategia de fusión:")
print("  1. results ← drivers (inner join en driverId)")
print("  2. resultados ← races (inner join en raceId)")
print("  3. resultados ← circuits (inner join en circuitId)")
print("  4. resultados ← pit_stops (left join en raceId, driverId)")
print("  5. resultados ← qualifying (left join en raceId, driverId)")
print("  6. resultados ← lap_times (left join en raceId, driverId)")
print("\nJustificación:")
print("  • Inner joins (1-3): Garantizan integridad de datos básicos")
print("  • Left joins (4-6): Preservan registros aunque falten detalles de carrera")

print("\n[4.1] Realizando fusión step-by-step...")

# Paso 1: results ← drivers
filas_iniciales = len(results)
data = results.merge(drivers, on='driverId', how='inner')
print(f"  Step 1: results ← drivers")
print(f"    {filas_iniciales:>8} → {len(data):>8} filas (inner join)")

# Paso 2: ← races
filas_anteriores = len(data)
data = data.merge(races, on='raceId', how='inner')
print(f"  Step 2: ↑ + races")
print(f"    {filas_anteriores:>8} → {len(data):>8} filas (inner join)")

# Paso 3: ← circuits
filas_anteriores = len(data)
data = data.merge(circuits, on='circuitId', how='inner')
print(f"  Step 3: ↑ + circuits")
print(f"    {filas_anteriores:>8} → {len(data):>8} filas (inner join)")

# Paso 4: ← pit_stops (left join)
filas_anteriores = len(data)
data = data.merge(pit_stops, on=['raceId', 'driverId'], how='left', 
                 suffixes=('', '_pit_stop'))
print(f"  Step 4: ↑ + pit_stops")
print(f"    {filas_anteriores:>8} → {len(data):>8} filas (left join)")

# Paso 5: ← qualifying (left join)
filas_anteriores = len(data)
data = data.merge(qualifying, on=['raceId', 'driverId'], how='left', 
                 suffixes=('', '_quali'))
print(f"  Step 5: ↑ + qualifying")
print(f"    {filas_anteriores:>8} → {len(data):>8} filas (left join)")

# Paso 6: ← lap_times (left join)
filas_anteriores = len(data)
data = data.merge(lap_times, on=['raceId', 'driverId'], how='left',
                 suffixes=('', '_lap'))
print(f"  Step 6: ↑ + lap_times")
print(f"    {filas_anteriores:>8} → {len(data):>8} filas (left join)")

print("\n[4.2] Analizando multiplicidad post-fusión...")
print(f"  Filas totales después de fusión: {len(data):,}")
print("\n  Explicación de la multiplicidad N:M:")
print("  • Una carrera/piloto tiene múltiples filas debido a:")
print("    - Múltiples pit_stops (una por parada)")
print("    - Múltiples lap_times (una por vuelta)")
print("  • Esto puede causar over-fitting si no se maneja adecuadamente")
print("  • Los modelos ML pueden inferir relaciones por vuelta/parada")

# ============================================================================
# SECCIÓN 5: SELECCIÓN DE CARACTERÍSTICAS Y TARGET
# ============================================================================

print("\n" + "="*90)
print("[5/7] SELECCIÓN DE CARACTERÍSTICAS Y VARIABLE OBJETIVO")
print("="*90)

# Características numéricas seleccionadas
features_numericas = ['age', 'grid', 'fastestLapSpeed', 
                     'milliseconds_pit_stop', 'milliseconds_lap_time']

# Características categóricas seleccionadas
features_categoricas = ['nationality', 'constructorId']

# Lista completa de features (antes de encoding)
features_base = features_numericas + features_categoricas

print("\nFeatures numéricas seleccionadas:")
for feat in features_numericas:
    info = FEATURES_DEFINICION.get(feat, {})
    print(f"  • {feat:<20} | {info.get('unidad', 'N/A'):<15} | {info.get('descripcion', 'No disponible')[:50]}")

print("\nFeatures categóricas seleccionadas:")
for feat in features_categoricas:
    info = FEATURES_DEFINICION.get(feat, {})
    print(f"  • {feat:<20} | {info.get('cardinalidad', 'N/A'):<15} | {info.get('descripcion', 'No disponible')[:50]}")

# Crear variable objetivo
print("\n[5.1] Generando variable objetivo (win)...")
data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
victorias_count = data['win'].sum()
total_count = len(data)
victorias_pct = (victorias_count / total_count) * 100

print(f"  • Total observaciones: {total_count:,}")
print(f"  • Victorias: {victorias_count:,} ({victorias_pct:.2f}%)")
print(f"  • Sin victorias: {total_count - victorias_count:,} ({100 - victorias_pct:.2f}%)")
print(f"  • Imbalance ratio: 1:{(total_count - victorias_count) / victorias_count:.1f}")

# ============================================================================
# SECCIÓN 6: LIMPIEZA FINAL: ELIMINACIÓN DE NAs
# ============================================================================

print("\n" + "="*90)
print("[6/7) LIMPIEZA FINAL: ELIMINACIÓN DE OBSERVACIONES INCOMPLETAS")
print("="*90)

# Filtrar filas con NAs en features o target
filas_antes = len(data)
data = data.dropna(subset=features_base + ['win'])
filas_despues = len(data)
filas_eliminadas = filas_antes - filas_despues

print(f"\n[6.1] Eliminando filas con valores faltantes en features o target...")
print(f"  Filas antes de limpieza: {filas_antes:,}")
print(f"  Filas después de limpieza: {filas_despues:,}")
print(f"  Filas eliminadas: {filas_eliminadas:,} ({(filas_eliminadas/filas_antes*100):.2f}%)")
print(f"  Porcentaje restante: {(filas_despues/filas_antes*100):.2f}%")

# ============================================================================
# SECCIÓN 7: CODIFICACIÓN ONE-HOT PARA VARIABLES CATEGÓRICAS
# ============================================================================

print("\n" + "="*90)
print("[7/7] CODIFICACIÓN ONE-HOT (PREPARACIÓN FINAL)")
print("="*90)

# Preparar X e y
X = data[features_base].copy()
y = data['win'].copy()

print(f"\n[7.1] Dataset antes de one-hot encoding:")
print(f"  Observaciones: {len(X):,}")
print(f"  Features: {X.shape[1]}")

# Aplicar one-hot encoding
print(f"\n[7.2] Aplicando one-hot encoding a variables categóricas...")
X_encoded = pd.get_dummies(X, columns=features_categoricas, dtype=float)

print(f"\n[7.3] Dataset después de one-hot encoding:")
print(f"  Observaciones: {len(X_encoded):,}")
print(f"  Features totales: {X_encoded.shape[1]}")
print(f"  Features numéricas originales: {len(features_numericas)}")
print(f"  Features one-hot de nationality: {len([c for c in X_encoded.columns if c.startswith('nationality_')])}")
print(f"  Features one-hot de constructorId: {len([c for c in X_encoded.columns if c.startswith('constructorId_')])}")

# ============================================================================
# SECCIÓN 8: GUARDADO DEL DATASET FINAL
# ============================================================================

print("\n" + "="*90)
print("[8/8] GUARDANDO DATASET FINAL")
print("="*90)

# Combinar features y target en un solo DataFrame
dataset_final = X_encoded.copy()
dataset_final['win'] = y.values

# Guardar dataset principal
nombre_archivo_dataset = 'dataset_tesis_f1.csv'
dataset_final.to_csv(nombre_archivo_dataset, index=False)

print(f"\n[8.1] Dataset principal guardado en: {nombre_archivo_dataset}")
print(f"  • Filas: {len(dataset_final):,}")
print(f"  • Columnas: {len(dataset_final.columns)}")

# ============================================================================
# SECCIÓN 9: GENERACIÓN DE METADATOS DEL DATASET
# ============================================================================

print("\n" + "="*90)
print("[9/9] GENERANDO METADATOS COMPLETOS")
print("="*90)

# Crear DataFrame de metadatos
metadata_list = []

#Agregar metadatos de configuración
metadata_list.append({
    'tipo': 'configuracion',
    'clave': 'version',
    'valor': CONFIG['version_dataset']
})
metadata_list.append({
    'tipo': 'configuracion',
    'clave': 'fecha_generacion',
    'valor': CONFIG['fecha']
})
metadata_list.append({
    'tipo': 'configuracion',
    'clave': 'descripcion',
    'valor': CONFIG['descripcion']
})

# Agregar metadatos de las características originales
for feature, info in FEATURES_DEFINICION.items():
    metadata_list.append({
        'tipo': 'feature_original',
        'clave': feature,
        'descripcion': info['descripcion'],
        'fuente': info['fuente'],
        'transformacion': info['transformacion'],
        'tipo_dato': info['tipo'],
        'unidad': info.get('unidad', 'N/A'),
        'cardinalidad': info.get('cardinalidad', info.get('rango', 'N/A'))
    })

# Agregar metadatos de la variable objetivo
for target, info in TARGET_DEFINICION.items():
    metadata_list.append({
        'tipo': 'target',
        'clave': target,
        'descripcion': info['descripcion'],
        'fuente': info['fuente'],
        'transformacion': info['transformacion'],
        'tipo_dato': info['tipo'],
        'valores': info['valores'],
        'balance_clases': info['balance']
    })

# Agregar metadatos del proceso de fusión
metadata_list.append({
    'tipo': 'fusion',
    'clave': 'filas_antes_fusion',
    'valor': filas_iniciales
})
metadata_list.append({
    'tipo': 'fusion',
    'clave': 'filas_despues_fusion',
    'valor': len(data)
})
metadata_list.append({
    'tipo': 'fusion',
    'clave': 'metodo_fusion',
    'valor': '6 joins: 3 inner + 3 left'
})
metadata_list.append({
    'tipo': 'fusion',
    'clave': 'multiplicidad',
    'valor': 'N:M debido a pit_stops y lap_times'
})

# Agregar metadatos de limpieza
metadata_list.append({
    'tipo': 'limpieza',
    'clave': 'filas_antes_limpieza',
    'valor': filas_antes
})
metadata_list.append({
    'tipo': 'limpieza',
    'clave': 'filas_despues_limpieza',
    'valor': filas_despues
})
metadata_list.append({
    'tipo': 'limpieza',
    'clave': 'filas_eliminadas',
    'valor': filas_eliminadas
})
metadata_list.append({
    'tipo': 'limpieza',
    'clave': 'porcentaje_eliminado',
    'valor': f"{(filas_eliminadas/filas_antes*100):.2f}%"
})

# Agregar metadatos de estadísticas de clases
metadata_list.append({
    'tipo': 'estadisticas_clases',
    'clave': 'total_observaciones',
    'valor': total_count
})
metadata_list.append({
    'tipo': 'estadisticas_clases',
    'clave': 'victorias',
    'valor': victorias_count
})
metadata_list.append({
    'tipo': 'estadisticas_clases',
    'clave': 'sin_victorias',
    'valor': total_count - victorias_count
})
metadata_list.append({
    'tipo': 'estadisticas_clases',
    'clave': 'porcentaje_victorias',
    'valor': f"{victorias_pct:.2f}%"
})
metadata_list.append({
    'tipo': 'estadisticas_clases',
    'clave': 'balance_ratio',
    'valor': f"1:{(total_count - victorias_count) / victorias_count:.1f}"
})

# Agregar metadatos de archivos fuente
for i, dinfo in enumerate(datasets_info, 1):
    metadata_list.append({
        'tipo': 'archivo_fuente',
        'clave': f'archivo_{i}',
        'nombre': dinfo['archivo'],
        'filas': dinfo['filas'],
        'columnas': dinfo['columnas']
    })

# Agregar metadatos de features del dataset final
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'nombre_archivo',
    'valor': nombre_archivo_dataset
})
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'observaciones',
    'valor': len(dataset_final)
})
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'columnas_totales',
    'valor': len(dataset_final.columns)
})
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'features_numericas',
    'valor': len(features_numericas)
})
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'features_nationality',
    'valor': len([c for c in dataset_final.columns if c.startswith('nationality_')])
})
metadata_list.append({
    'tipo': 'dataset_final',
    'clave': 'features_constructorid',
    'valor': len([c for c in dataset_final.columns if c.startswith('constructorId_')])
})

# Guardar metadatos en CSV
metadata_df = pd.DataFrame(metadata_list)
nombre_archivo_metadata = 'metadata_dataset.csv'
metadata_df.to_csv(nombre_archivo_metadata, index=False)

print(f"\n[9.1] Metadatos guardados en: {nombre_archivo_metadata}")
print(f"  • Total registros de metadatos: {len(metadata_list)}")

# ============================================================================
# SECCIÓN 10: GENERACIÓN DE INFORME DE ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

print("\n" + "="*90)
print("[10/10] GENERANDO REPORTE DE ESTADÍSTICAS")
print("="*90)

# Estadísticas del dataset final
estadisticas = {
    'metrica': [
        'Total observaciones',
        'Total victorias',
        'Total sin victorias',
        'Porcentaje victorias',
        'Porcentaje sin victorias',
        'Ratio desbalance',
        'Features totales',
        'Instancias por observación (promedio)'
    ],
    'valor': [
        len(dataset_final),
        int(dataset_final['win'].sum()),
        int((1 - dataset_final['win']).sum()),
        f"{dataset_final['win'].mean()*100:.2f}%",
        f"{(1 - dataset_final['win']).mean()*100:.2f}%",
        f"1:{dataset_final[dataset_final['win'] == 1].shape[0] / max(1, (dataset_final[dataset_final['win'] == 0].shape[0])):.1f}",
        len(dataset_final.columns) - 1,  # -1 para excluir el target
        f"{dataset_final[dataset_final.columns[:-1]].notna().sum().mean():.1f}"
    ],
    'comentario': [
        'Número total de observaciones en el dataset',
        'Casos positivos (piloto ganó)',
        'Casos negativos (piloto no ganó)',
        'Porcentaje de casos positivos',
        'Porcentaje de casos negativos',
        'Cantidad de casos negativos por cada positivo',
        'Variable predictoras (excluyendo target)',
        'Promedio de valores no-nulos por fila'
    ]
}

estadisticas_df = pd.DataFrame(estadisticas)
nombre_archivo_estadisticas = 'estadisticas_dataset.csv'
estadisticas_df.to_csv(nombre_archivo_estadisticas, index=False)

print(f"\n[10.1] Estadísticas descriptivas guardadas en: {nombre_archivo_estadisticas}")

# ============================================================================
# SECCIÓN 11: GENERACIÓN DE ARCHIVO DE DOCUMENTACIÓN MARKDOWN
# ============================================================================

print("\n" + "="*90)
print("[11/11] GENERANDO DOCUMENTACIÓN MARKDOWN")
print("="*90)

doc_contenido = f"""# Documentación del Dataset: dataset_tesis_f1.csv

## 1. Descripción General

Este dataset constituye la base de datos unificada para el desarrollo de modelos 
predictivos de victorias en carreras de Fórmula 1. Se ha generado a partir de 
la integración y transformación de 8 archivos CSV originales mediante un pipeline 
estándarizado de procesamiento de datos.

### 1.1 Identificación del Dataset
- **Nombre del archivo**: `dataset_tesis_f1.csv`
- **Versión**: {CONFIG['version_dataset']}
- **Fecha de generación**: {CONFIG['fecha']}
- **Propósito**: Base de datos compartida para entrenamiento de tres modelos ML

### 1.2 Dimensiones del Dataset
- **Observaciones**: {len(dataset_final):,}
- **Variables**: {len(dataset_final.columns)} (incluyendo target)
- **Variables predictoras**: {len(dataset_final.columns) - 1}
- **Variable objetivo**: 1 (`win`)

---

## 2. Variables del Dataset

### 2.1 Variable Objetivo

| Variable | Descripción | Tipo | Valores | Balance |
|----------|-------------|------|---------|---------|
| `win` | El piloto ganó la carrera | Binaria | 0=No, 1=Sí | {victorias_pct:.2f}% positivos |

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
| nationality | Nacionalidad del piloto | {len([c for c in dataset_final.columns if c.startswith('nationality_')])} países | One-hot (nationality_COUNTRY) |
| constructorId | Constructor/escudería | {len([c for c in dataset_final.columns if c.startswith('constructorId_')])} escuderías | One-hot (constructorId_N) |

---

## 3. Proceso de Generación del Dataset

### 3.1 Archivos Fuente

Se utilizaron 8 archivos CSV originales:

1. **circuits.csv** - Información de circuitos ({len(circuits)} filas)
2. **results.csv** - Resultados de carreras ({len(results)} filas)
3. **drivers.csv** - Información de pilotos ({len(drivers)} filas)
4. **pit_stops.csv** - Paradas en boxes ({len(pit_stops)} filas)
5. **races.csv** - Catálogo de carreras ({len(races)} filas)
6. **qualifying.csv** - Datos de clasificación ({len(qualifying)} filas)
7. **lap_times.csv** - Tiempos de vuelta ({len(lap_times)} filas)
8. **constructors.csv** - Datos de constructores ({len(constructors)} filas)

### 3.2 Pipeline de Preprocesamiento

#### Paso 1: Limpieza de Valores Nulos
- Sustitución de '\\N' (formato PostgreSQL) por `NaN` (valor nulo pandas)
- Aplicado a: results, drivers, pit_stops, races, qualifying, lap_times

#### Paso 2: Ingeniería de Características
- Cálculo de edad actual desde fecha de nacimiento: `(fecha_actual - dob) / 365`
- Renombrado de columnas ambiguas:
  - pit_stops: `milliseconds` → `milliseconds_pit_stop`
  - lap_times: `milliseconds` → `milliseconds_lap_time`

#### Paso 3: Fusión de Datasets (6 JOINs)

| Paso | Operación | Tipo | Claves | Resultado |
|------|-----------|------|--------|-----------|
| 1 | results ← drivers | inner | driverId | {filas_iniciales} → {len(data)} + (N/A) |
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
- Filas eliminadas: {filas_eliminadas:,} ({(filas_eliminadas/filas_antes*100):.2f}%)
- Filas retenidas: {filas_despues:,} ({(filas_despues/filas_antes*100):.2f}%)

#### Paso 5: Codificación One-Hot
- Transformación de variables categóricas a dummy variables
- `nationality` → {len([c for c in dataset_final.columns if c.startswith('nationality_')])} columnas binarias
- `constructorId` → {len([c for c in dataset_final.columns if c.startswith('constructorId_')])} columnas binarias
- Resultado: {len(dataset_final.columns)} columnas totales

---

## 4. Estadísticas del Dataset

### 4.1 Distribución de la Variable Objetivo

| Métrica | Valor |
|---------|-------|
| Total observaciones | {len(dataset_final):,} |
| Victorias (positivos) | {victorias_count:,} ({victorias_pct:.2f}%) |
| Sin victorias (negativos) | {total_count - victorias_count:,} ({100 - victorias_pct:.2f}%) |
| Ratio desbalance | 1:{(total_count - victorias_count) / victorias_count:.1f} {': (negativos por cada positivo)' if victorias_count > 0 else ': Datos insuficientes'} |

### 4.2 Composición del Dataset

| Tipo de Feature | Categoría | Cantidad |
|-----------------|-----------|----------|
| Numéricas | Variables continuas/discretas | {len(features_numericas)} |
| Categóricas | nationality (one-hot) | {len([c for c in dataset_final.columns if c.startswith('nationality_')])} |
| Categóricas | constructorId (one-hot) | {len([c for c in dataset_final.columns if c.startswith('constructorId_')])} |
| **Total Features** | - | {len(dataset_final.columns) - 1} |

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
**Fecha**: {CONFIG['fecha']}  
**Versión del documento**: 1.0
"""

nombre_archivo_doc = 'DOCUMENTACION_DATASET_TESIS.md'
with open(nombre_archivo_doc, 'w', encoding='utf-8') as f:
    f.write(doc_contenido)

print(f"\n[11.1] Documentación guardada en: {nombre_archivo_doc}")

# ============================================================================
# SECCIÓN 12: RESUMEN FINAL
# ============================================================================

print("\n" + "="*90)
print(" RESUMEN FINAL DEL PROCESO")
print("="*90)

print(f"\n✓ Dataset final generado: dataset_tesis_f1.csv")
print(f"  - Observaciones: {len(dataset_final):,}")
print(f"  - Variables: {len(dataset_final.columns)}")
print(f"  - Victorias: {victorias_count:,} ({victorias_pct:.2f}%)")

print(f"\n✓ Archivos de metadatos generados:")
print(f"  1. metadata_dataset.csv ({len(metadata_list)} registros)")
print(f"  2. estadisticas_dataset.csv ({len(estadisticas)} registros)")
print(f"  3. DOCUMENTACION_DATASET_TESIS.md (documentación completa)")

print("\n✓ Proceso completado exitosamente")
print("\nEste dataset ahora puede ser utilizado por los tres modelos:")
print("  • modelo_regresion_lineal.py")
print("  • modelo_random_forest.py")
print("  • modelo_perceptron_multicapa.py")

print("\n" + "="*90)
print(" PROCESO TERMINADO")
print("="*90)
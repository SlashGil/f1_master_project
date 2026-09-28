"""
====================================================
PREPARACIÓN DEL DATASET PARA MODELOS DE PREDICCIÓN
====================================================

Script de preparación de datos para Tesis
Propósito: Crear datasets limpios y documentados para
           regresión lineal, random forest y perceptrón multicapa

Metodología:
- Ingeniería de características rigurosa
- Documentación de cada transformación
- Generación de datasets procesados y sin procesar
- Garantía de reproducibilidad académica

Autor: Proyecto Tesis F1 Predictive
Última actualización: 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# CONFIGURACIÓN Y DIRECTORIOS
# ============================================================================

OUTPUT_DIR = "datasets_procesados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("SISTEMA DE PREPARACIÓN DE DATASET PARA TESIS")
print("="*80)
print(f"\nDirectorio de salida: {os.path.abspath(OUTPUT_DIR)}/")

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS FUENTE
# ============================================================================

print("\n" + "="*80)
print("FASE 1: CARGA DE DATOS ORIGINALES")
print("="*80)

print("\nCargando datasets de Fórmula 1...")

# Definición manual de tipos para asegurar consistencia
dtypes = {
    'resultId': 'int64', 'raceId': 'int64', 'driverId': 'int64',
    'constructorId': 'float64', 'number': 'float64', 'grid': 'float64',
    'position': 'object', 'positionText': 'object', 'positionOrder': 'int64',
    'points': 'float64', 'laps': 'int64', 'time': 'object',
    'milliseconds': 'float64', 'fastestLap': 'float64',
    'rank': 'object', 'fastestLapTime': 'object', 'fastestLapSpeed': 'float64',
    'statusId': 'int64'
}

try:
    circuits = pd.read_csv('circuits.csv')
    results = pd.read_csv('results.csv', dtype=dtypes, low_memory=False)
    drivers = pd.read_csv('drivers.csv')
    pit_stops = pd.read_csv('pit_stops.csv')
    races = pd.read_csv('races.csv')
    qualifying = pd.read_csv('qualifying.csv')
    lap_times = pd.read_csv('lap_times.csv')
    constructors = pd.read_csv('constructors.csv')
    
    print("\n✓ Datasets cargados exitosamente:")
    print(f"  - circuits.csv:        {circuits.shape[0]:>7} filas")
    print(f"  - results.csv:          {results.shape[0]:>7} filas")
    print(f"  - drivers.csv:         {drivers.shape[0]:>7} filas")
    print(f"  - pit_stops.csv:       {pit_stops.shape[0]:>7} filas")
    print(f"  - races.csv:           {races.shape[0]:>7} filas")
    print(f"  - qualifying.csv:      {qualifying.shape[0]:>7} filas")
    print(f"  - lap_times.csv:       {lap_times.shape[0]:>7} filas")
    print(f"  - constructors.csv:     {constructors.shape[0]:>7} filas")
    
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("\nPor favor, ubique este script en el directorio con los archivos CSV")
    exit(1)

# ============================================================================
# SECCIÓN 2: LIMPIEZA Y NORMALIZACIÓN
# ============================================================================

print("\n" + "="*80)
print("FASE 2: LIMPIEZA Y NORMALIZACIÓN DE DATOS")
print("="*80)

datasets_to_clean = {
    'results': results,
    'drivers': drivers,
    'pit_stops': pit_stops,
    'races': races,
    'qualifying': qualifying,
    'lap_times': lap_times
}

for name, df in datasets_to_clean.items():
    # Reemplazar '\N' (formato PostgreSQL NULL) por NaN
    filas_antes = df.shape[0]
    df.replace('\\N', np.nan, inplace=True)
    
    # Contar cuántos valores fueron reemplazados
    valores_nan = df.isna().sum().sum()
    print(f"  ✓ {name:<15}: {valores_nan:>6} valores nulos detectados y normalizados")

print("\nNormalización de fecha de nacimiento de pilotos:")
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
total_validas = drivers['dob'].notna().sum()
print(f"  • Fechas válidas: {total_validas} de {len(drivers)} pilotos")

# Renombrado de columnas ambiguas para evitar colisiones en fusión
columnas_renombradas = {
    'pit_stops': {'milliseconds': 'milliseconds_pit_stop'},
    'pit_stops': {'duration': 'duration_pit_stop'},
    'lap_times': {'milliseconds': 'milliseconds_lap_time'}
}

for df_name, mapping in columnas_renombradas.items():
    if df_name == 'pit_stops':
        pit_stops.rename(columns={'milliseconds': 'milliseconds_pit_stop'}, inplace=True)
    elif df_name == 'lap_times':
        lap_times.rename(columns={'milliseconds': 'milliseconds_lap_time'}, inplace=True)

print("\n✓ Columnas ambiguas renombradas:")
print(f"  - pit_stops.milliseconds → milliseconds_pit_stop")
print(f"  - lap_times.milliseconds   → milliseconds_lap_time")

# ============================================================================
# SECCIÓN 3: INGENIERÍA DE CARACTERÍSTICAS
# ============================================================================

print("\n" + "="*80)
print("FASE 3: INGENIERÍA DE CARACTERÍSTICAS")
print("="*80)

print("\n[3.1] Cálculo de edad de pilotos (derivado temporal)")
drivers['age'] = drivers['dob'].apply(
    lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else np.nan
)
edades_validas = drivers['age'].notna().sum()
print(f"  ✓ Edad calculada para {edades_validas} pilotos")
print(f"  ✓ Rango de edad: {drivers['age'].min():.0f} - {drivers['age'].max():.0f} años")

print("\n[3.2] Agregación de tiempos de vuelta por carrera")
# Para reducir multiplicidad, calcularemos estadísticas agregadas por carrera/piloto
lap_stats = lap_times.groupby(['raceId', 'driverId']).agg({
    'milliseconds_lap_time': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
lap_stats.columns = ['raceId', 'driverId', 
                     'lap_time_mean', 'lap_time_std', 'lap_time_min', 
                     'lap_time_max', 'laps_completed']
print(f"  ✓ Estadísticas de vueltas calculadas para {lap_stats.shape[0]} carrera/piloto combinaciones")

print("\n[3.3] Agregación de pit stops por carrera")
pit_stats = pit_stops.groupby(['raceId', 'driverId']).agg({
    'milliseconds_pit_stop': ['mean', 'std', 'count']
}).reset_index()
pit_stats.columns = ['raceId', 'driverId',
                     'pit_stop_mean', 'pit_stop_std', 'pit_stops_count']
print(f"  ✓ Estadísticas de pit stops calculadas para {pit_stats.shape[0]} carrera/piloto combinaciones")

print("\n[3.4] Agregación de datos de clasificación")
quali_stats = qualifying.groupby(['raceId', 'driverId']).agg({
    'q1': 'mean', 'q2': 'mean', 'q3': 'mean',
    'position': 'first'
}).reset_index()
quali_stats.columns = ['raceId', 'driverId', 'quali_q1_mean', 'quali_q2_mean',
                      'quali_q3_mean', 'quali_position']
print(f"  ✓ Datos de clasificación agregados para {quali_stats.shape[0]} carrera/piloto combinaciones")

# ============================================================================
# SECCIÓN 4: FUSIÓN DE DATASETS (JOIN)
# ============================================================================

print("\n" + "="*80)
print("FASE 4: FUSIÓN DE DATASETS")
print("="*80)

print("\nEjecutando fusión secuencial de datasets...")

# Fusion central: results como tabla principal
print("\n[4.1] results ← drivers (inner join on driverId)")
data = results.merge(drivers, on='driverId')
print(f"  ✓ Registros: {data.shape[0]}")

print("\n[4.2] resultados ← races (inner join on raceId)")
data = data.merge(races, on='raceId')
print(f"  ✓ Registros: {data.shape[0]}")

print("\n[4.3] resultados ← circuits (inner join on circuitId)")
data = data.merge(circuits, on='circuitId')
print(f"  ✓ Registros: {data.shape[0]}")

print("\n[4.4] resultados ← constructors (inner join on constructorId)")
data = data.merge(constructors, on='constructorId')
print(f"  ✓ Registros: {data.shape[0]}")

# Fusiones con tablas agregadas
print("\n[4.5] resultados ← lap_stats (left join)")
data = data.merge(lap_stats, on=['raceId', 'driverId'], how='left')
print(f"  ✓ Registros: {data.shape[0]}")

print("\n[4.6] resultados ← pit_stats (left join)")
data = data.merge(pit_stats, on=['raceId', 'driverId'], how='left')
print(f"  ✓ Registros: {data.shape[0]}")

print("\n[4.7] resultados ← quali_stats (left join)")
data = data.merge(quali_stats, on=['raceId', 'driverId'], how='left')
print(f"  ✓ Registros: {data.shape[0]}")

print(f"\n✓ Dataset fusionado final: {data.shape[0]} filas, {data.shape[1]} columnas")

# ============================================================================
# SECCIÓN 5: DEFINICIÓN DE VARIABLES PARA MODELOS
# ============================================================================

print("\n" + "="*80)
print("FASE 5: SELECCIÓN Y DEFINICIÓN DE VARIABLES")
print("="*80)

print("\n[5.1] Variables predictoras seleccionadas:")

# Variables numéricas
numeric_vars = [
    'age',                    # Edad del piloto
    'grid',                   # Posición de salida (clasificación)
    'laps_completed',         # Vueltas completadas en carrera
    'lap_time_mean',          # Tiempo medio por vuelta
    'lap_time_std',           # Desviación estándar de vueltas
    'lap_time_min',           # Mejor vuelta
    'pit_stop_mean',          # Tiempo medio de pit stop
    'pit_stop_std',           # Variabilidad de pit stops
    'pit_stops_count',       # Número de pit stops
    'quali_q1_mean',         # Tiempo medio Q1
    'quali_q2_mean',         # Tiempo medio Q2
    'quali_q3_mean',         # Tiempo medio Q3
    'quali_position'         # Posición en clasificación
]

# Variables categóricas
categorical_vars = [
    'nationality',           # Nacionalidad del piloto
    'constructorId'          # Constructor/escudería
]

print("\nVariables numéricas (12):")
for var in numeric_vars:
    if var in data.columns:
        n_valid = data[var].notna().sum()
        pct_availability = (n_valid / len(data)) * 100
        print(f"  • {var:<25} | {n_valid:>6}/{len(data):<6} datos ({pct_availability:>5.1f}% disponibles)")

print("\nVariables categóricas (2):")
for var in categorical_vars:
    if var in data.columns:
        n_unique = data[var].nunique()
        print(f"  • {var:<25} | {n_unique:>6} categorías únicas")

print("\n[5.2] Variable objetivo (Target):")

# Variable objetivo: Victoria del piloto (binaria)
data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

total_samples = len(data)
victorias = data['win'].sum()
porcentaje = (victorias / total_samples) * 100

print(f"  Variable: win (binaria)")
print(f"    • 0 = No ganó")
print(f"    • 1 = Ganó")
print(f"  • Victorias totales: {victorias} de {total_samples} ({porcentaje:.2f}%)")
print(f"  • Dataset desbalanceado (esperado en predicción de victorias)")

# ============================================================================
# SECCIÓN 6: FILTRADO Y LIMPIEZA FINAL
# ============================================================================

print("\n" + "="*80)
print("FASE 6: LIMPIEZA FINAL DE DATASET")
print("="*80)

print("\n[6.1] Filtrando muestras completas...")

# Verificar disponibilidad de variables
selected_vars = numeric_vars + categorical_vars
vars_disponibles = [v for v in selected_vars if v in data.columns]
print(f"  • Variables disponibles: {len(vars_disponibles)}/{len(selected_vars)}")

# Crear dataset limpio solo con variables seleccionadas
data_clean = data[['win'] + vars_disponibles].dropna()

print(f"\n[6.2] Estadísticas del dataset limpio:")
print(f"  • Filas originales: {len(data)}")
print(f"  • Filas tras limpieza: {len(data_clean)}")
print(f"  • Filas eliminadas (NA): {len(data) - len(data_clean)} ({(1 - len(data_clean)/len(data))*100:.1f}%)")

# Verificación de balance de clases en dataset limpio
victorias_limpias = data_clean['win'].sum()
porcentaje_limpias = (victorias_limpias / len(data_clean)) * 100
print(f"\n[6.3] Distribución de clases en dataset limpio:")
print(f"  • Victorias: {victorias_limpias} ({porcentaje_limpias:.2f}%)")
print(f"  • No victorias: {len(data_clean) - victorias_limpias} ({100 - porcentaje_limpias:.2f}%)")

# ============================================================================
# SECCIÓN 7: GENERACIÓN DE DATASETS DE SALIDA
# ============================================================================

print("\n" + "="*80)
print("FASE 7: GENERACIÓN DE DATASETS PROCESADOS")
print("="*80)

# Dataset 1: Raw (sin procesar adicional)
output_raw = f"{OUTPUT_DIR}/f1_dataset_raw.csv"
data_clean.to_csv(output_raw, index=False)
mb_size = os.path.getsize(output_raw) / (1024 * 1024)
print(f"\n✓ Dataset crítico generado: {output_raw}")
print(f"  Tamaño: {mb_size:.2f} MB")
print(f"  Filas: {len(data_clean)}")
print(f"  Columnas: {data_clean.shape[1]}")

# Dataset 2: Procesado con one-hot encoding
print("\nAplicando one-hot encoding a variables categóricas...")
data_processed = pd.get_dummies(data_clean, columns=['nationality', 'constructorId'], dtype=float)

output_processed = f"{OUTPUT_DIR}/f1_dataset_processed.csv"
data_processed.to_csv(output_processed, index=False)
mb_size = os.path.getsize(output_processed) / (1024 * 1024)
print(f"✓ Dataset procesado generado: {output_processed}")
print(f"  Tamaño: {mb_size:.2f} MB")
print(f"  Filas: {len(data_processed)}")
print(f"  Columnas: {data_processed.shape[1]} (expansión por one-hot encoding)")

# ============================================================================
# SECCIÓN 8: DOCUMENTACIÓN DEL DATASET
# ============================================================================

print("\n" + "="*80)
print("FASE 8: GENERACIÓN DE DOCUMENTACIÓN")
print("="*80)

doc_filename = f"{OUTPUT_DIR}/DOCUMENTACION_DATASET.txt"
with open(doc_filename, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DOCUMENTACIÓN DEL DATASET DE FÓRMULA 1 PARA TESIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. DESCRIPCIÓN GENERAL\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"Este dataset contiene datos históricos de carreras de Fórmula 1 procesados para\n")
    f.write(f"el entrenamiento de modelos predictivos de victorias.\n\n")
    f.write(f"• Fecha de creación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"• Filas totales: {len(data_clean)}\n")
    f.write(f"• Muestras sin NAs: {len(data_clean)}\n")
    f.write(f"• Victorias: {victorias_limpias} ({porcentaje_limpias:.2f}%)\n\n")
    
    f.write("2. ARCHIVOS GENERADOS\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"A. f1_dataset_raw.csv\n")
    f.write(f"   - Dataset sin procesar adicional\n")
    f.write(f"   - Variables numéricas + variables categóricas originales\n")
    f.write(f"   - Adecuado para modelos que manejan categorías nativamente\n\n")
    f.write(f"B. f1_dataset_processed.csv\n")
    f.write(f"   - Dataset con one-hot encoding aplicado\n")
    f.write(f"   - Variables numéricas + variables dummy (0/1)\n")
    f.write(f"   - Adecuado para regresión lineal y perceptrón multicapa\n")
    f.write(f"   - Columnas expandidas de {data_clean.shape[1]} a {data_processed.shape[1]}\n\n")
    
    f.write("3. VARIABLES DEL DATASET (f1_dataset_raw.csv)\n")
    f.write("-" * 80 + "\n\n")
    
    f.write("[Variable Objetivo]\n")
    f.write(f"win                : Binaria (0=No ganó, 1=Ganó)\n\n")
    
    f.write("[Variables Numéricas]\n")
    for var in numeric_vars:
        if var in data_clean.columns:
            desc = {
                'age': 'Edad del piloto en años (derivado de fecha de nacimiento)',
                'grid': 'Posición de salida en la carrera (1=primera posición)',
                'laps_completed': 'Número de vueltas completadas por el piloto',
                'lap_time_mean': 'Tiempo promedio por vuelta en milisegundos',
                'lap_time_std': 'Desviación estándar de tiempos de vuelta',
                'lap_time_min': 'Mejor tiempo de vuelta en milisegundos',
                'pit_stop_mean': 'Tiempo promedio de paradas en boxes (milisegundos)',
                'pit_stop_std': 'Desviación estándar de tiempos de pit stops',
                'pit_stops_count': 'Número total de paradas en boxes',
                'quali_q1_mean': 'Tiempo promedio en sesión Q1 de clasificación',
                'quali_q2_mean': 'Tiempo promedio en sesión Q2 de clasificación',
                'quali_q3_mean': 'Tiempo promedio en sesión Q3 de clasificación',
                'quali_position': 'Posición final en clasificación'
            }
            f.write(f"{var:<25} : {desc.get(var, 'Sin descripción')}\n")
    
    f.write("\n[Variables Categóricas]\n")
    f.write(f"nationality       : Nacionalidad del piloto (~50 categorías únicas)\n")
    f.write(f"constructorId     : Identificador del constructor (~210 constructores)\n\n")
    
    f.write("4. TRANSFORMACIONES APLICADAS\n")
    f.write("-" * 80 + "\n\n")
    f.write("[A. Limpieza de Nulos]\n")
    f.write(f"- Sustitución de valores '\\N' (PostgreSQL null) por NaN\n")
    f.write(f"- Eliminación de filas con valores faltantes en variables seleccionadas\n\n")
    f.write(f"[B. Ingeniería de Características]\n")
    f.write(f"- Cálculo de edad a partir de fecha de nacimiento (dob → age)\n")
    f.write(f"- Agregación de tiempos de vuelta (media, std, min, max, count)\n")
    f.write(f"- Agregación de pit stops (media, std, count)\n")
    f.write(f"- Agregación de datos de clasificación (media Q1/Q2/Q3, posición)\n")
    f.write(f"- Renombrado de columnas ambiguas para evitar colisiones en fusión\n\n")
    f.write(f"[C. Fusión de Datasets]\n")
    f.write(f"- Joins secuenciales de 7 datasets originales\n")
    f.write(f"- Preservación de integridad referencial (joins inner para datos clave)\n")
    f.write(f"- Joins left para tablas agregadas (preservar muestras completas)\n\n")
    f.write(f"[D. Procesamiento Adicional]\n")
    f.write(f"- (Solo en f1_dataset_processed.csv) One-hot encoding de variables categóricas\n")
    f.write(f"- Nacionalidad → variables dummy (nationality_British, nationality_German, etc.)\n")
    f.write(f"- ConstructorId → variables dummy (constructorId_1, constructorId_2, etc.)\n\n")
    
    f.write("5. PROPORCIÓN DE CLASES (Imbalance)\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"Clase 0 (No victorias): {len(data_clean) - victorias_limpias} muestras ({100 - porcentaje_limpias:.2f}%)\n")
    f.write(f"Clase 1 (Victorias)    : {victorias_limpias} muestras      ({porcentaje_limpias:.2f}%)\n\n")
    f.write(f"NOTA: El desbalance de clases es esperado y debe considerarse durante\n")
    f.write(f"      el entrenamiento de modelos (e.g., class_weight='balanced', SMOTE)\n\n")
    
    f.write("6. FUENTES DE DATOS ORIGINALES\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"Los datos provienen de fuentes históricas de Fórmula 1 (1950-presente):\n\n")
    f.write(f"• circuits.csv        : Circuitos donde se han realizado carreras\n")
    f.write(f"• results.csv         : Resultados detallados de carreras\n")
    f.write(f"• drivers.csv         : Información demográfica de pilotos\n")
    f.write(f"• pit_stops.csv       : Registros de paradas en boxes\n")
    f.write(f"• races.csv           : Catálogo de carreras\n")
    f.write(f"• qualifying.csv     : Datos de sesiones de clasificación\n")
    f.write(f"• lap_times.csv       : Tiempos por vuelta\n")
    f.write(f"• constructors.csv    : Información de constructores/escuderías\n\n")
    
    f.write("7. RECOMENDACIONES DE USO\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"[Para Regresión Lineal]      → Usar f1_dataset_processed.csv\n")
    f.write(f"[Para Perceptrón Multicapa]  → Usar f1_dataset_processed.csv\n")
    f.write(f"[Para Random Forest]         → Usar f1_dataset_raw.csv (maneja categorías nativamente)\n\n")
    f.write(f"Para todos los modelos:\n")
    f.write(f"- Aplicar StandardScaler a variables numéricas antes del entrenamiento\n")
    f.write(f"- Considerar técnicas de balanceo de clases (class_weight, SMOTE)\n")
    f.write(f"- Para predicción 'a priori' (antes de carrera): usar solo grid, age,\n")
    f.write(f"  nationality, constructorId\n")
    f.write(f"- Para predicción 'a posteriori' (durante carrera): usar todas las variables\n")
    f.write(f"  incluy lap_time_mean, pit_stop_mean, etc.\n\n")
    
    f.write("8. LIMITACIONES CONOCIDAS\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"1. Fuga Temporal: positionOrder se usa para derivar 'win' y como\n")
    f.write(f"   característica (se recomienda remover para predicción a priori)\n\n")
    f.write(f"2. Desbalance de Clases: ~5% victorias vs 95% no victorias.\n")
    f.write(f"   Puede sesgar modelos hacia predicción de 'no ganar'\n\n")
    f.write(f"3. Variabilidad Temporal: El rendimiento constructores cambia por temporada.\n")
    f.write(f"   Modelos pueden no generalizar a temporadas futuras.\n\n")
    f.write(f"4. Datos Faltantes: Se utiliza eliminación de NAs, lo que reduce el\n")
    f.write(f"   tamaño del dataset y puede introducir sesgo de selección.\n\n")
    
    f.write("="*80 + "\n")
    f.write("FIN DE DOCUMENTACIÓN\n")
    f.write("="*80 + "\n")

print(f"✓ Documentación generada: {doc_filename}")

# ============================================================================
# SECCIÓN 9: RESUMEN ESTADÍSTICO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

print("\n\nARCHIVOS GENERADOS:")
print("-" * 80)
print(f"1. {OUTPUT_DIR}/f1_dataset_raw.csv")
print(f"   - Dataset sin procesar adicional")
print(f"   - {len(data_clean)} filas, {data_clean.shape[1]} columnas")
print(f"   - Variables numéricas + categóricas originales")
print(f"\n2. {OUTPUT_DIR}/f1_dataset_processed.csv")
print(f"   - Dataset con one-hot encoding aplicado")
print(f"   - {len(data_processed)} filas, {data_processed.shape[1]} columnas (expandidas)")
print(f"   - Variables numéricas + dummy variables")
print(f"\n3. {OUTPUT_DIR}/DOCUMENTACION_DATASET.txt")
print(f"   - Documentación completa de variables y transformaciones")

print("\n\nCARACTERÍSTICAS DEL DATASET:")
print("-" * 80)
print(f"Variable Objetivo    : win (binaria)")
print(f"Variables Numéricas  : {len([v for v in numeric_vars if v in data_clean.columns])}")
print(f"Variables Categóricas: {len([v for v in categorical_vars if v in data_clean.columns])}")
print(f"Victorias            : {victorias_limpias} ({porcentaje_limpias:.2f}%)")
print(f"No Victorias         : {len(data_clean) - victorias_limpias} ({100-porcentaje_limpias:.2f}%)")
print(f"Ratio clases         : 1:{(len(data_clean)-victorias_limpias)/victorias_limpias:.1f}")

print("\n\nRECOMENDACIONES:")
print("-" * 80)
print("✓ Regresión Lineal      → Usar f1_dataset_processed.csv")
print("✓ Perceptrón Multicapa  → Usar f1_dataset_processed.csv")
print("✓ Random Forest         → Usar f1_dataset_raw.csv")
print("✓ Para todos → Aplicar StandardScaler antes del entrenamiento")
print("✓ Para todos → Considerar class_weight='balanced' para imbalance")

print("\n" + "="*80)
print("PREPARACIÓN DE DATASET COMPLETADA")
print("="*80)
print("\nLos datasets procesados están listos para ser utilizados por los modelos")
print("de aprendizaje automático en su tesis académica.\n")

print("\nIMPORTANTE: Este script puede ejecutarse nuevamente para regenerar los datasets")
print("           con modificaciones en el preprocesamiento o ingeniería de características.\n")
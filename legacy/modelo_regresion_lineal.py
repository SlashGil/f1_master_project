"""
Modelo 1: Regresión Lineal para Análisis de Importancia de Variables
======================================================================

Este script implementa un modelo de regresión lineal para predecir la probabilidad
de victoria en carreras de Fórmula 1 y determina qué variables son más determinantes.

Autor: Proyecto F1 Predictive
Objetivo: Analizar importancia de variables mediante coeficientes de regresión
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*70)
print("MODELO 1: REGRESIÓN LINEAL - Análisis de Importancia de Variables")
print("="*70)

# ============================================================================
# SECCIÓN 1: Carga de Datos
# ============================================================================

print("\n[1/7] Cargando datasets...")


circuits = pd.read_csv('circuits.csv')
results = pd.read_csv('results.csv')
drivers = pd.read_csv('drivers.csv')
pit_stops = pd.read_csv('pit_stops.csv')
races = pd.read_csv('races.csv')
qualifying = pd.read_csv('qualifying.csv')
lap_times = pd.read_csv('lap_times.csv')
constructors = pd.read_csv('constructors.csv')

print(f"✓ Circuitos: {circuits.shape}")
print(f"✓ Resultados: {results.shape}")
print(f"✓ Pilotos: {drivers.shape}")
print(f"✓ Pit Stops: {pit_stops.shape}")
print(f"✓ Carreras: {races.shape}")
print(f"✓ Clasificación: {qualifying.shape}")
print(f"✓ Tiempos de Vuelta: {lap_times.shape}")
print(f"✓ Constructores: {constructors.shape}")

# ============================================================================
# SECCIÓN 2: Preprocesamiento de Datos
# ============================================================================

print("\n[2/7] Aplicando preprocesamiento...")

# Sustituir '\N' por NaN
results.replace('\\N', np.nan, inplace=True)
drivers.replace('\\N', np.nan, inplace=True)
pit_stops.replace('\\N', np.nan, inplace=True)
races.replace('\\N', np.nan, inplace=True)
qualifying.replace('\\N', np.nan, inplace=True)
lap_times.replace('\\N', np.nan, inplace=True)

# Calcular edad de pilotos
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
drivers['age'] = drivers['dob'].apply(lambda x: (datetime.now() - x).days // 365 if pd.notnull(x) else np.nan)

# Renombrar columnas ambiguas
pit_stops.rename(columns={'milliseconds': 'milliseconds_pit_stop'}, inplace=True)
lap_times.rename(columns={'milliseconds': 'milliseconds_lap_time'}, inplace=True)

print("✓ Sustitución de valores nulos completada")
print("✓ Cálculo de edad de pilotos completado")
print("✓ Renombrado de columnas ambiguas completado")

# ============================================================================
# SECCIÓN 3: Fusión de Datasets
# ============================================================================

print("\n[3/7] Fusionando datasets...")

data = results.merge(drivers, on='driverId') \
              .merge(races, on='raceId') \
              .merge(circuits, on='circuitId') \
              .merge(pit_stops, on=['raceId', 'driverId'], how='left') \
              .merge(qualifying, on=['raceId', 'driverId'], how='left') \
              .merge(lap_times, on=['raceId', 'driverId'], how='left')

print(f"✓ Dataset fusionado creado: {data.shape[0]} filas, {data.shape[1]} columnas")

# ============================================================================
# SECCIÓN 4: Selección de Características y Variable Objetivo
# ============================================================================

print("\n[4/7) Seleccionando características y generando target...")

# Variables predictoras
features = ['age', 'nationality', 'constructorId', 'grid', 
            'fastestLapSpeed', 'milliseconds_pit_stop', 'milliseconds_lap_time']

# Variable objetivo
target = 'win'

# Crear variable binaria de victoria
data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)

# Eliminar filas con valores nulos en las características
data = data.dropna(subset=features + [target])

print(f"✓ Dataset limpio: {data.shape[0]} filas tras eliminar NAs")
print(f"✓ Distribución de clases - Victorias: {data[target].sum()} ({data[target].mean()*100:.2f}%)")

# ============================================================================
# SECCIÓN 5: Codificación de Variables Categóricas
# ============================================================================

print("\n[5/7] Codificando variables categóricas...")

X = data[features].copy()
y = data[target]

# One-hot encoding para variables categóricas
X = pd.get_dummies(X, columns=['nationality', 'constructorId'], dtype=float)

print(f"✓ Variables después de encoding: {X.shape[1]} features")

# ============================================================================
# SECCIÓN 6: División y Escalado de Datos
# ============================================================================

print("\n[6/7] Dividiendo datos y aplicando escalado...")

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Entrenamiento: {X_train.shape[0]} muestras")
print(f"✓ Prueba: {X_test.shape[0]} muestras")

# ============================================================================
# SECCIÓN 7: Entrenamiento de Regresión Lineal
# ============================================================================

print("\n[7/7] Entrenando modelo de Regresión Lineal...")
print("-" * 70)

# Crear y entrenar modelo
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predicciones
y_train_pred = linear_model.predict(X_train_scaled)
y_test_pred = linear_model.predict(X_test_scaled)

# ============================================================================
# SECCIÓN 8: Evaluación del Modelo
# ============================================================================

print("\n" + "="*70)
print("EVALUACIÓN DEL MODELO DE REGRESIÓN LINEAL")
print("="*70)

# Métricas de entrenamiento
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f"\nMétricas de Entrenamiento:")
print(f"  • MSE (Error Cuadrático Medio): {train_mse:.6f}")
print(f"  • R² (Coeficiente de Determinación): {train_r2:.6f}")
print(f"  • MAE (Error Absoluto Medio): {train_mae:.6f}")

# Métricas de prueba
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nMétricas de Prueba:")
print(f"  • MSE: {test_mse:.6f}")
print(f"  • R²: {test_r2:.6f}")
print(f"  • MAE: {test_mae:.6f}")

# ============================================================================
# SECCIÓN 9: Análisis de Importancia de Variables (Coeficientes)
# ============================================================================

print("\n" + "="*70)
print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
print("="*70)

# Obtener coeficientes y sus valores absolutos
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_model.coef_,
    'Abs_Coefficient': np.abs(linear_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

# Top 20 variables más importantes
top_features = coefficients.head(20)
print(f"\nTop 20 Variables Más Determinantes (por magnitud de coeficiente):")
print("-"*70)
for idx, row in top_features.iterrows():
    coef_sign = "+" if row['Coefficient'] >= 0 else " "
    print(f"  {row['Feature']:<50} | {coef_sign}{row['Coefficient']:>12.4f}")

# Guardar análisis completo
coefficients.to_csv('feature_importance_linear_regression.csv', index=False)
print(f"\n✓ Análisis de importancia guardado en: feature_importance_linear_regression.csv")

# ============================================================================
# SECCIÓN 10: Visualización de Resultados
# ============================================================================

print("\nGenerando visualizaciones...")

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')

# Figura 1: Top 20 variables por importancia
plt.figure(figsize=(12, 8))
colors = ['red' if c < 0 else 'green' for c in top_features['Coefficient']]
plt.barh(range(len(top_features)), top_features['Abs_Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Magnitud del Coeficiente (Importancia Absoluta)')
plt.title('Top 20 Variables Más Determinantes para Ganar\n(Regresión Lineal)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_linear1.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 guardado: feature_importance_linear1.png")

# Figura 2: Coeficientes con signo
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(top_features)), top_features['Coefficient'], 
                color=['red' if c < 0 else 'green' for c in top_features['Coefficient']])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Valor del Coeficiente')
plt.title('Top 20 Variables - Coeficiente Con Signo\n(Verde=Positivo, Rojo=Negativo)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_linear2.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 guardado: feature_importance_linear2.png")

# Figura 3: Predicciones vs Valores Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valor Real (Victoria 0=No, 1=Sí)')
plt.ylabel('Predicción del Modelo')
plt.title('Predicciones vs Valores Reales - Regresión Lineal')
plt.tight_layout()
plt.savefig('predictions_vs_actual_linear.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 guardado: predictions_vs_actual_linear.png")

# ============================================================================
# SECCIÓN 11: Resumen Comparativo
# ============================================================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO: VARIABLES MÁS DETERMINANTES")
print("="*70)

print(f"\nLa variable MÁS determinante es: **{top_features.iloc[0]['Feature']}**")
print(f"   Coeficiente: {top_features.iloc[0]['Coefficient']:.4f}")
print(f"   Interpretación: {'Aumenta significativamente la probabilidad de victoria' if top_features.iloc[0]['Coefficient'] > 0 else 'Reduce significativamente la probabilidad de victoria'}")

print(f"\nLas 5 variables más importantes son:")
for i in range(min(5, len(top_features))):
    row = top_features.iloc[i]
    effect = "POSITIVO" if row['Coefficient'] > 0 else "NEGATIVO"
    print(f"  {i+1}. {row['Feature']:<45} | {effect}")
    print(f"     Coeficiente: {row['Coefficient']:.4f}")

# Convertir IDs de constructores a nombres donde sea posible
constructor_features = [f for f in top_features['Feature'] if 'constructorId_' in f]
if constructor_features:
    print(f"\n📊 Constructores más influyentes en las predicciones:")
    for feat in constructor_features[:5]:
        const_id = feat.split('_')[1]
        name = constructors[constructors['constructorId'] == int(const_id)]['name']
        if len(name) > 0:
            print(f"  • Constructor ID {const_id} ({name.iloc[0]})")

print("\n" + "="*70)
print("PROCESO COMPLETADO")
print("="*70)
print(f"✓ Modelo entrenado correctamente")
print(f"✓ Análisis de importancia de variables completado")
print(f"✓ Visualizaciones generadas")
print(f"✓ Archivos guardados en directorio actual")
print("\nPara ejecutar este script, asegúrese de estar en el directorio con todos los archivos CSV")
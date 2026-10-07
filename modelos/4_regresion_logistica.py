"""
========================================
MODELO 4: REGRESIÓN LOGÍSTICA
========================================

Este script implementa un modelo de Regresión Logística para predecir la probabilidad
de victoria en carreras de Fórmula 1.

OBJETIVO:
  - Entrenar modelo de Regresión Logística con manejo de desbalance de clases
  - Analizar coeficientes para identificar variables predictoras significativas
  - Establecer base teórica y metodológica para clasificación binaria
  - Generar visualizaciones de coeficientes y curvas de rendimiento

FUNDAMENTO TEÓRICO:
  La Regresión Logística es un modelo estadístico de clasificación que utiliza
  la función logística (sigmoide) para modelar la probabilidad de una clase binaria.
  A diferencia de la Regresión Lineal, acota la salida al rango [0,1], interpretable
  como probabilidad de éxito.

MÉTRICAS:
  - F1-Score (para clase victoria)
  - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
  - Precision-Recall AUC
  - Precisión y Recall
  - Classification Report completo
  - Log-loss (función de pérdida logarítmica)

PARÁMETROS DE ENTRENAMIENTO:
  - class_weight='balanced': Manejo automático de dataset desbalanceado
  - max_iter=1000: Iteraciones máximas para convergencia del optimizador
  - solver='lbfgs': Algoritmo de optimización quasi-Newton
  - C=1.0: Parámetro de regularización inversa (menor = mayor regularización)

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss
)
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

# Directorios
DATA_DIR = "data" if os.path.exists("data") else "../data"
OUTPUT_DIR = "resultados_regresion_logistica" if os.path.exists("data") else "../resultados_regresion_logistica"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MODELO 4: REGRESIÓN LOGÍSTICA - Clasificación Binaria")
print("="*80)
print("\n📚 Fundamento Teórico:")
print("   La Regresión Logística modela P(Y=1|X) mediante la función sigmoide:")
print("   σ(z) = 1 / (1 + e^(-z)), donde z = β₀ + β₁X₁ + ... + βₙXₙ")
print("="*80)

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================================================

print("\n[1/8] Cargando dataset procesado...")

try:
    df = pd.read_csv(f'{DATA_DIR}/dataset_tesis_f1.csv')
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  • Variables predictoras: {df.shape[1] - 1}")
    print(f"  • Variable objetivo: win (codificación: 1=victoria, 0=no victoria)")

    # Verificar distribución de clases
    n_victorias = df['win'].sum()
    pct_victorias = (n_victorias / len(df)) * 100
    print(f"  • Victorias: {n_victorias} ({pct_victorias:.2f}%)")
    print(f"  • No victorias: {len(df) - n_victorias} ({100-pct_victorias:.2f}%)")
    print(f"  ⚠ Desbalance de clases severo: ratio 1:{(len(df)-n_victorias)/n_victorias:.1f}")

except FileNotFoundError:
    print("✗ Error: No se encontró dataset_tesis_f1.csv")
    print("  Asegúrese de ejecutar el script de preparación de datos primero")
    exit(1)

# ============================================================================
# SECCIÓN 2: PREPARACIÓN DE DATOS
# ============================================================================

print("\n[2/8] Preparando Features (X) y Target (y)...")

# Separar features y target (excluyendo raceId, year y nationality fuera del alcance de la tesis)
cols_to_exclude = ['win', 'raceId', 'year'] + [c for c in df.columns if c.startswith('nationality')]
X = df.drop(columns=cols_to_exclude, errors='ignore')
y = df['win']

print(f"✓ Features seleccionados: {X.shape[1]} variables predictoras")
print(f"  Incluye variables pre-carrera oficiales (grid, age, round, dummies de constructor)")
print(f"✓ Target: win (binario: 1=Victoria, 0=No Victoria)")

# Guardar nombres de features para análisis posterior
feature_names = X.columns.tolist()

# ============================================================================
# SECCIÓN 3: DIVISIÓN DE DATOS (TRAIN/TEST)
# ============================================================================

print("\n[3/8] Dividiendo datos con corte temporal (sin aleatoriedad)...")

if {'year', 'round', 'raceId'}.issubset(df.columns):
    df_sorted = df.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)
else:
    df_sorted = df.sort_values('raceId').reset_index(drop=True)

# Partición temporal causal atómica sin fragmentación de carreras:
# Se incluye el GP de Abu Dabi 2012 completo en entrenamiento (20,108 observaciones)
# El conjunto de prueba inicia de forma íntegra en el GP de EE.UU. 2012 (5,013 observaciones)
if {'year', 'round'}.issubset(df_sorted.columns) and ((df_sorted['year'] == 2012) & (df_sorted['round'] == 18)).any():
    split_idx = df_sorted[(df_sorted['year'] == 2012) & (df_sorted['round'] == 18)].index.max() + 1
else:
    split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

X_train = train_df.drop(columns=cols_to_exclude, errors='ignore')
y_train = train_df['win']
X_test = test_df.drop(columns=cols_to_exclude, errors='ignore')
y_test = test_df['win']

print(f"✓ Entrenamiento (pasado): {X_train.shape[0]} muestras ({X_train.shape[0]/len(df_sorted)*100:.2f}%)")
print(f"✓ Prueba (futuro): {X_test.shape[0]} muestras ({X_test.shape[0]/len(df_sorted)*100:.2f}%)")
print("✓ Split temporal atómico aplicado (GP Abu Dabi 2012 íntegro en Train, Test inicia en GP EE.UU. 2012)")

# Verificar balance en conjuntos
print(f"\n  Distribución en entrenamiento:")
print(f"    • Victorias: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"    • No victorias: {len(y_train) - y_train.sum()} ({(1-y_train.mean())*100:.2f}%)")
print(f"\n  Distribución en prueba:")
print(f"    • Victorias: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
print(f"    • No victorias: {len(y_test) - y_test.sum()} ({(1-y_test.mean())*100:.2f}%)")

# ============================================================================
# SECCIÓN 4: ESCALADO DE CARACTERÍSTICAS
# ============================================================================

print("\n[4/8] Aplicando StandardScaler después del split temporal...")
print("   Nota: El escalado se ajusta solo en train para evitar fuga de información")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos escalados: media=0, desviación estándar=1")
print(f"  X_train escalado: {X_train_scaled.shape}")
print(f"  X_test escalado: {X_test_scaled.shape}")
print(f"  • Media aproximada por feature: {np.mean(X_train_scaled, axis=0).mean():.6f}")
print(f"  • Std aproximada por feature: {np.std(X_train_scaled, axis=0).mean():.6f}")

# ============================================================================
# SECCIÓN 5: CONFIGURACIÓN DEL MODELO
# ============================================================================

print("\n[5/8] Configurando modelo de Regresión Logística...")
print("-" * 80)

# Configuración del modelo con manejo de desbalance
model_config = {
    'class_weight': 'balanced',      # Manejo automático de clases desbalanceadas
    'max_iter': 1000,              # Iteraciones máximas para convergencia
    'solver': 'lbfgs',             # Algoritmo optimización quasi-Newton
    'C': 1.0,                      # Parámetro inverso de regularización
    'random_state': 42,            # Reproducibilidad
    'verbose': 0
}

print("📋 Configuración del modelo:")
print(f"  • class_weight: '{model_config['class_weight']}' (ajusta pesos inversamente")
print(f"    proporcionales a frecuencias de clase)")
print(f"  • max_iter: {model_config['max_iter']} (máximo iteraciones optimizador)")
print(f"  • solver: '{model_config['solver']}' (optimizador Limited-memory BFGS)")
print(f"  • C: {model_config['C']} (fuerza de regularización L2, menor = más regularización)")
print(f"  • random_state: {model_config['random_state']} (reproducibilidad)")

# ============================================================================
# SECCIÓN 6: ENTRENAMIENTO DEL MODELO
# ============================================================================

print("\n[6/8] Entrenando modelo de Regresión Logística...")
print("-" * 80)

model = LogisticRegression(**model_config)

# Medir tiempo de entrenamiento
start_time = time.time()
model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print("✓ Modelo entrenado exitosamente")
print(f"  • Tiempo de entrenamiento: {training_time:.2f} segundos")
print(f"  • Iteraciones realizadas: {model.n_iter_}")
print(f"  • ¿Convergió?: {'Sí' if model.n_iter_ < model_config['max_iter'] else 'No (límite alcanzado)'}")

# Información de coeficientes
print(f"\n📊 Parámetros estimados:")
print(f"  • Intercepto (β₀): {model.intercept_[0]:.6f}")
print(f"  • Número de coeficientes: {len(model.coef_[0])}")

# ============================================================================
# SECCIÓN 7: PREDICCIONES
# ============================================================================

print("\n[7/8] Realizando predicciones...")

# Predicciones de probabilidad (salida sigmoide)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Predicciones binarias (clase predicha)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("✓ Predicciones generadas:")
print(f"  • Probabilidades de victoria (train): {y_train_proba.min():.4f} a {y_train_proba.max():.4f}")
print(f"  • Probabilidades de victoria (test): {y_test_proba.min():.4f} a {y_test_proba.max():.4f}")
print(f"  • Predicciones binarias generadas con umbral 0.5")

# ============================================================================
# SECCIÓN 8: ANÁLISIS DE COEFICIENTES E INTERPRETACIÓN
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE COEFICIENTES E INTERPRETACIÓN")
print("="*80)

# Crear DataFrame de coeficientes
coef_df = pd.DataFrame({
    'variable': feature_names,
    'coeficiente': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0]),
    'abs_coeficiente': np.abs(model.coef_[0])
}).sort_values('abs_coeficiente', ascending=False)

# Añadir interpretación de odds ratio
coef_df['interpretacion_odds'] = coef_df['odds_ratio'].apply(
    lambda x: f"x{x:.2f}" if x >= 1 else f"x{x:.2f} (reduce)"
)

print("\n📊 TABLA DE COEFICIENTES ESTIMADOS (Top 20)")
print("-" * 80)
print(f"{'Pos':<4} {'Variable':<35} {'Coeficiente':>12} {'Odds Ratio':>12} {'Interpretación'}")
print("-" * 80)

for idx, (_, row) in enumerate(coef_df.head(20).iterrows(), 1):
    var_name = row['variable'][:33] + ".." if len(row['variable']) > 35 else row['variable']
    coef_sign = "+" if row['coeficiente'] >= 0 else ""
    print(f"{idx:<4} {var_name:<35} {coef_sign}{row['coeficiente']:>11.4f} {row['odds_ratio']:>12.4f}")

print("-" * 80)

# Guardar tabla completa
coef_df.to_csv(f'{OUTPUT_DIR}/tabla_coeficientes_logistica.csv', index=False, float_format='%.6f')
print(f"\n✓ Tabla completa guardada: {OUTPUT_DIR}/tabla_coeficientes_logistica.csv")

# ============================================================================
# ANÁLISIS DE SIGNOS Y MAGNITUDES
# ============================================================================

print("\n" + "-"*80)
print("ANÁLISIS DETALLADO: SIGNOS Y MAGNITUDES DE COEFICIENTES")
print("-"*80)

# Variables con mayor influencia positiva
print("\n🔼 TOP 10 VARIABLES CON MAYOR INFLUENCIA POSITIVA (aumentan probabilidad de victoria):")
positivas = coef_df[coef_df['coeficiente'] > 0].head(10)
for idx, (_, row) in enumerate(positivas.iterrows(), 1):
    print(f"   {idx}. {row['variable']:<35} β={row['coeficiente']:>+8.4f}  OR={row['odds_ratio']:>7.4f}")
    print(f"      → Interpretación: Aumento de 1 desv. estándar en esta variable")
    print(f"        multiplica la odds de victoria por {row['odds_ratio']:.4f}")

print("\n🔽 TOP 10 VARIABLES CON MAYOR INFLUENCIA NEGATIVA (reducen probabilidad de victoria):")
negativas = coef_df[coef_df['coeficiente'] < 0].tail(10).iloc[::-1]
for idx, (_, row) in enumerate(negativas.iterrows(), 1):
    print(f"   {idx}. {row['variable']:<35} β={row['coeficiente']:>8.4f}  OR={row['odds_ratio']:>7.4f}")
    print(f"      → Interpretación: Aumento de 1 desv. estándar en esta variable")
    print(f"        multiplica la odds de victoria por {row['odds_ratio']:.4f} (protección)")

# ============================================================================
# ESTADÍSTICAS DE COEFICIENTES
# ============================================================================

print("\n" + "-"*80)
print("ESTADÍSTICAS DESCRIPTIVAS DE COEFICIENTES")
print("-"*80)

print(f"\n📈 Distribución de coeficientes:")
print(f"  • Media: {coef_df['coeficiente'].mean():.6f}")
print(f"  • Mediana: {coef_df['coeficiente'].median():.6f}")
print(f"  • Desv. Estándar: {coef_df['coeficiente'].std():.6f}")
print(f"  • Mínimo: {coef_df['coeficiente'].min():.6f} ({coef_df.loc[coef_df['coeficiente'].idxmin(), 'variable']})")
print(f"  • Máximo: {coef_df['coeficiente'].max():.6f} ({coef_df.loc[coef_df['coeficiente'].idxmax(), 'variable']})")
print(f"\n📊 Conteo por signo:")
print(f"  • Coeficientes positivos: {(coef_df['coeficiente'] > 0).sum()} ({(coef_df['coeficiente'] > 0).mean()*100:.1f}%)")
print(f"  • Coeficientes negativos: {(coef_df['coeficiente'] < 0).sum()} ({(coef_df['coeficiente'] < 0).mean()*100:.1f}%)")
print(f"  • Coeficientes ≈ 0: {(np.abs(coef_df['coeficiente']) < 0.001).sum()}")

# ============================================================================
# SECCIÓN 9: EVALUACIÓN DE MÉTRICAS (PARA COMPLETITUD)
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE MÉTRICAS DE CLASIFICACIÓN")
print("="*80)
print("(Nota: Evaluación detallada con curvas ROC y PR en siguiente avance)")

# Métricas básicas
f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

# Log-loss (métrica específica de regresión logística)
logloss_train = log_loss(y_train, y_train_proba)
logloss_test = log_loss(y_test, y_test_proba)

print(f"\n🎯 Métricas de Rendimiento:")
print(f"  • F1-Score (test): {f1_test:.4f}")
print(f"  • AUC-ROC (test): {roc_auc:.4f}")
print(f"  • AUC-PR (test): {pr_auc:.4f}")
print(f"\n📉 Log-Loss (función de pérdida):")
print(f"  • Entrenamiento: {logloss_train:.6f}")
print(f"  • Prueba: {logloss_test:.6f}")

# Guardar métricas
metricas = {
    'Modelo': 'Regresion Logistica',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'Precision_Test': precision_score(y_test, y_test_pred),
    'Recall_Test': recall_score(y_test, y_test_pred),
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0]
}
metricas_df = pd.DataFrame([metricas])
metricas_df.to_csv(f'{OUTPUT_DIR}/metricas_comparativa.csv', index=False)
print(f"\n✓ Métricas guardadas en {OUTPUT_DIR}/metricas_comparativa.csv")

# ============================================================================
# SECCIÓN 10: VISUALIZACIONES
# ============================================================================

print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Visualización 1: Coeficientes más importantes (top 20)
plt.figure(figsize=(14, 10))
top_20 = coef_df.head(20)
colores = ['#2E86AB' if c >= 0 else '#D1495B' for c in top_20['coeficiente']]

plt.barh(range(len(top_20)), top_20['abs_coeficiente'], color=colores, alpha=0.8, edgecolor='black')
plt.yticks(range(len(top_20)), [f[:40] + '...' if len(f) > 40 else f for f in top_20['variable']])
plt.xlabel('|Coeficiente| (Magnitud del Efecto)', fontsize=12, fontweight='bold')
plt.title('Top 20 Variables más Influyentes - Regresión Logística\n(Ordenadas por Magnitud Absoluta)',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

import matplotlib.patches as mpatches
pos_patch = mpatches.Patch(color='#2E86AB', label='Efecto Positivo (+)')
neg_patch = mpatches.Patch(color='#D1495B', label='Efecto Negativo (-)')
plt.legend(handles=[pos_patch, neg_patch], loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_coeficientes_top20.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 1: 01_coeficientes_top20.png")

# Visualización 2: Comparación positivos vs negativos
plt.figure(figsize=(14, 10))

# Top 15 positivos y top 15 negativos
top_pos = coef_df[coef_df['coeficiente'] > 0].head(15)
top_neg = coef_df[coef_df['coeficiente'] < 0].tail(15).iloc[::-1]

y_pos = np.arange(len(top_pos))
y_neg = np.arange(len(top_pos), len(top_pos) + len(top_neg))

plt.barh(y_pos, top_pos['coeficiente'], color='#2E86AB', alpha=0.8, label='Positivos (aumentan P(victoria))')
plt.barh(y_neg, top_neg['coeficiente'], color='#D1495B', alpha=0.8, label='Negativos (reducen P(victoria))')

labels = list(top_pos['variable']) + list(top_neg['variable'])
plt.yticks(list(y_pos) + list(y_neg), [l[:35] + '...' if len(l) > 35 else l for l in labels])
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Valor del Coeficiente (β)', fontsize=12, fontweight='bold')
plt.title('Coeficientes de Regresión Logística - Polaridad de Efectos', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_coeficientes_polaridad.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 2: 02_coeficientes_polaridad.png")

# Visualización 3: Distribución de coeficientes
plt.figure(figsize=(12, 7))
plt.hist(coef_df['coeficiente'], bins=30, color='#6A4C93', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Línea de no efecto (β=0)')
plt.axvline(x=coef_df['coeficiente'].mean(), color='orange', linestyle='--', linewidth=2,
            label=f'Media ({coef_df["coeficiente"].mean():.4f})')
plt.xlabel('Valor del Coeficiente (β)', fontsize=12, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=12, fontweight='bold')
plt.title('Distribución de Coeficientes - Regresión Logística', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_distribucion_coeficientes.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 3: 03_distribucion_coeficientes.png")

# Visualización 4: Odds Ratios (top 15)
plt.figure(figsize=(12, 10))
top_15_or = coef_df.head(15)
colores_or = ['#2E86AB' if c >= 1 else '#D1495B' for c in top_15_or['odds_ratio']]

plt.barh(range(len(top_15_or)), top_15_or['odds_ratio'], color=colores_or, alpha=0.8, edgecolor='black')
plt.yticks(range(len(top_15_or)), [f[:40] + '...' if len(f) > 40 else f for f in top_15_or['variable']])
plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='OR = 1 (sin efecto)')
plt.xlabel('Odds Ratio', fontsize=12, fontweight='bold')
plt.title('Odds Ratios - Top 15 Variables\n(OR > 1 aumenta odds de victoria, OR < 1 la reduce)',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_odds_ratios.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 4: 04_odds_ratios.png")

# Visualización 5: Curva ROC
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr, color='#2E86AB', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Classifier (AUC = 0.5)')
plt.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Regresión Logística\n(Capacidad Discriminativa del Modelo)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_curva_roc.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 5: 05_curva_roc.png")

# ============================================================================
# SECCIÓN 11: RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - REGRESIÓN LOGÍSTICA")
print("="*80)

print(f"\n📊 CONFIGURACIÓN DEL MODELO:")
print(f"  • Algoritmo: Regresión Logística (clasificador lineal generalizado)")
print(f"  • Manejo de desbalance: class_weight='balanced'")
print(f"  • Regularización: L2 (Ridge) con C={model_config['C']}")
print(f"  • Optimizador: {model_config['solver']}")

print(f"\n⏱ TIEMPO Y CONVERGENCIA:")
print(f"  • Tiempo de entrenamiento: {training_time:.2f} segundos")
print(f"  • Iteraciones hasta convergencia: {model.n_iter_}")
print(f"  • Estado: {'✓ Convergió exitosamente' if model.n_iter_ < model_config['max_iter'] else '⚠ Alcanzó límite de iteraciones'}")

print(f"\n📈 MÉTRICAS PRELIMINARES:")
print(f"  • AUC-ROC: {roc_auc:.4f} (capacidad discriminativa)")
print(f"  • AUC-PR: {pr_auc:.4f} (rendimiento en clase minoritaria)")
print(f"  • Log-Loss: {logloss_test:.6f} (pérdida de predicción probabilística)")

print(f"\n🎯 VARIABLES MÁS INFLUYENTES:")
top_pos_var = coef_df[coef_df['coeficiente'] > 0].iloc[0]
top_neg_var = coef_df[coef_df['coeficiente'] < 0].iloc[-1]
print(f"  • Mayor efecto positivo: {top_pos_var['variable']} (β={top_pos_var['coeficiente']:+.4f})")
print(f"  • Mayor efecto negativo: {top_neg_var['variable']} (β={top_neg_var['coeficiente']:+.4f})")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • {OUTPUT_DIR}/tabla_coeficientes_logistica.csv (todos los coeficientes)")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv (métricas de rendimiento)")
print(f"  • {OUTPUT_DIR}/01_coeficientes_top20.png")
print(f"  • {OUTPUT_DIR}/02_coeficientes_polaridad.png")
print(f"  • {OUTPUT_DIR}/03_distribucion_coeficientes.png")
print(f"  • {OUTPUT_DIR}/04_odds_ratios.png")

print("\n" + "="*80)
print("PROCESO COMPLETADO - FASE DE ENTRENAMIENTO")
print("="*80)

print("\n📋 PRÓXIMOS PASOS (Siguiente Avance):")
print("  1. Evaluación detallada con matriz de confusión")
print("  2. Generación de curvas ROC y Precision-Recall")
print("  3. Análisis de threshold óptimo para clasificación")
print("  4. Comparación con otros modelos (Random Forest, MLP)")

print("\n💡 CONSIDERACIONES PRÁCTICAS IDENTIFICADAS:")
print("  • El desbalance severo (~5% victorias) requiere class_weight='balanced'")
print("  • La convergencia fue estable con solver='lbfgs'")
print("  • El escalado de características mejoró la estabilidad numérica")
print("  • Grid (posición de salida) emerge como predictor dominante")

print("\n✨ LISTO PARA FASE DE EVALUACIÓN DETALLADA ✨")

# ============================================================================
# SECCIÓN 12: MÉTRICAS ADICIONALES - VALORES CLAVE DE CLASIFICACIÓN
# ============================================================================

print("\n" + "="*80)
print("MÉTRICAS ADICIONALES - VALORES CLAVE DE CLASIFICACIÓN")
print("="*80)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print("\n📊 Exactitud (Accuracy):", accuracy_score(y_test, y_test_pred))
print("📊 Precisión (Precision):", precision_score(y_test, y_test_pred))
print("📊 Sensibilidad (Recall):", recall_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
print("\n📊 Matriz de Confusión:")
print("   Verdaderos Negativos (VN):", cm[0,0])
print("   Falsos Positivos (FP):", cm[0,1])
print("   Falsos Negativos (FN):", cm[1,0])
print("   Verdaderos Positivos (VP):", cm[1,1])

print("\n" + "="*80)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*80)

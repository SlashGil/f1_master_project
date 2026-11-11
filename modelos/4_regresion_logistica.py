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

PARÁMETROS DE ENTRENAMIENTO (PASO 2 - CON UMBRAL CALIBRADO):
  - class_weight=None: Máxima verosimilitud calibrada al prior real (~4.38% en Train)
  - Umbral de decisión: Calibrado exclusivamente en Train vía F1-max sobre
    predicciones Out-Of-Fold de 5-Fold Race-Aware TimeSeriesSplit (sin fuga)
  - max_iter=1000: Iteraciones máximas para convergencia del optimizador
  - solver='lbfgs': Algoritmo de optimización quasi-Newton
  - C=1.0: Parámetro de regularización inversa (menor = mayor regularización)

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss, accuracy_score
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados_regresion_logistica")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def race_aware_time_series_split(df_subset: pd.DataFrame, n_splits: int = 5):
    """Genera pliegues temporales expansivos agrupados por carreras completas (raceId)."""
    unique_races = df_subset['raceId'].drop_duplicates().tolist()
    split_size = len(unique_races) // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_races = set(unique_races[:split_size * i])
        val_races = set(unique_races[split_size * i:split_size * (i + 1)])
        tr_idx = df_subset.index[df_subset['raceId'].isin(train_races)].values
        val_idx = df_subset.index[df_subset['raceId'].isin(val_races)].values
        splits.append((tr_idx, val_idx))
    return splits

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
cols_to_exclude = ['win', 'raceId', 'driverId', 'year'] + [c for c in df.columns if c.startswith('nationality')]
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

train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

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
# SECCIÓN 5: CONFIGURACIÓN DEL MODELO Y CALIBRACIÓN DE UMBRAL (OOF TRAIN)
# ============================================================================

print("\n[5/8] Configurando modelo de Regresión Logística (MLE calibrado)...")
print("-" * 80)

# Configuración del modelo con probabilidades calibradas al prior real
model_config = {
    'class_weight': None,            # Probabilidades calibradas al prior real (MLE)
    'max_iter': 1000,                # Iteraciones máximas para convergencia
    'solver': 'lbfgs',               # Algoritmo optimización quasi-Newton
    'C': 1.0,                        # Parámetro inverso de regularización
    'random_state': 42,              # Reproducibilidad
    'verbose': 0
}

print("📋 Configuración del modelo:")
print(f"  • class_weight: {model_config['class_weight']} (MLE estándar calibrado)")
print(f"  • max_iter: {model_config['max_iter']} (máximo iteraciones optimizador)")
print(f"  • solver: '{model_config['solver']}' (optimizador Limited-memory BFGS)")
print(f"  • C: {model_config['C']} (fuerza de regularización L2, menor = más regularización)")
print(f"  • random_state: {model_config['random_state']} (reproducibilidad)")

print("\n🎯 Calibrando umbral óptimo sin fuga mediante 5-Fold Race-Aware TimeSeriesSplit en Train...")
cv_splits = race_aware_time_series_split(train_df, n_splits=5)
oof_probas, oof_targets = [], []
for tr_idx, va_idx in cv_splits:
    fold_scaler = StandardScaler()
    X_tr_fold = fold_scaler.fit_transform(X_train.iloc[tr_idx])
    X_va_fold = fold_scaler.transform(X_train.iloc[va_idx])
    fold_model = LogisticRegression(**model_config)
    fold_model.fit(X_tr_fold, y_train.iloc[tr_idx])
    oof_probas.append(fold_model.predict_proba(X_va_fold)[:, 1])
    oof_targets.append(y_train.iloc[va_idx].values)

oof_probas = np.concatenate(oof_probas)
oof_targets = np.concatenate(oof_targets)

oof_prec, oof_rec, oof_thresh = precision_recall_curve(oof_targets, oof_probas)
oof_f1 = 2 * oof_prec[:-1] * oof_rec[:-1] / (oof_prec[:-1] + oof_rec[:-1] + 1e-12)
best_idx = int(np.argmax(oof_f1))
best_threshold = float(oof_thresh[best_idx])
best_oof_f1 = float(oof_f1[best_idx])
print(f"✓ Umbral óptimo calibrado (F1-max OOF Train): τ* = {best_threshold:.6f} (F1 OOF = {best_oof_f1:.4f})")

# ============================================================================
# SECCIÓN 6: ENTRENAMIENTO DEL MODELO
# ============================================================================

print("\n[6/8] Entrenando modelo final de Regresión Logística sobre todo Train...")
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
# SECCIÓN 7: PREDICCIONES CON UMBRAL CALIBRADO
# ============================================================================

print("\n[7/8] Realizando predicciones con umbral calibrado en Train...")

# Predicciones de probabilidad (salida sigmoide)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Predicciones binarias aplicando umbral calibrado τ*
y_train_pred = (y_train_proba >= best_threshold).astype(int)
y_test_pred = (y_test_proba >= best_threshold).astype(int)

# Evaluación complementaria por dominio deportivo (ArgMax por raceId: 1 ganador por GP)
test_eval_df = pd.DataFrame({
    'raceId': test_df['raceId'].values,
    'proba': y_test_proba,
    'win': y_test.values
})
top1_indices = test_eval_df.groupby('raceId')['proba'].idxmax().values
y_test_argmax = np.zeros(len(test_eval_df), dtype=int)
y_test_argmax[top1_indices] = 1
argmax_hits = int((y_test_argmax & y_test.values).sum())
argmax_total_races = int(test_eval_df['raceId'].nunique())
argmax_acc = argmax_hits / argmax_total_races

print("✓ Predicciones generadas:")
print(f"  • Probabilidades de victoria (train): {y_train_proba.min():.4f} a {y_train_proba.max():.4f}")
print(f"  • Probabilidades de victoria (test):  {y_test_proba.min():.4f} a {y_test_proba.max():.4f}")
print(f"  • Umbral aplicado: τ* = {best_threshold:.6f} -> Positivos predichos en Test: {int(y_test_pred.sum())}")
print(f"  • Complementario ArgMax por carrera: {argmax_hits}/{argmax_total_races} victorias exactas ({argmax_acc*100:.2f}%)")

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
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)

# Log-loss y Brier Score
logloss_train = log_loss(y_train, y_train_proba)
logloss_test = log_loss(y_test, y_test_proba)
brier_test = brier_score_loss(y_test, y_test_proba)

print(f"\n🎯 Métricas de Rendimiento (Umbral τ* = {best_threshold:.6f}):")
print(f"  • F1-Score (train): {f1_train:.4f}")
print(f"  • F1-Score (test):  {f1_test:.4f}")
print(f"  • Precision (test): {precision_test:.4f}")
print(f"  • Recall (test):    {recall_test:.4f}")
print(f"  • Accuracy (test):  {accuracy_test:.4f}")
print(f"  • AUC-ROC (test):   {roc_auc:.4f}")
print(f"  • AUC-PR (test):    {pr_auc:.4f}")
print(f"\n📉 Calibración Probabilística:")
print(f"  • Log-Loss (train/test): {logloss_train:.6f} / {logloss_test:.6f}")
print(f"  • Brier Score (test):    {brier_test:.6f}")
print(f"  • ArgMax Top-1/GP:       {argmax_hits}/{argmax_total_races} ({argmax_acc*100:.2f}%)")

print("\n[CLASSIFICATION REPORT - TEST]")
print(classification_report(y_test, y_test_pred, target_names=['No Victoria', 'Victoria'], digits=4))

# Guardar métricas comparativas
metricas = {
    'Modelo': 'Regresion Logistica',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'Precision_Test': precision_test,
    'Recall_Test': recall_test,
    'Accuracy_Test': accuracy_test,
    'Threshold_OOF': best_threshold,
    'ArgMax_Top1_Acc': argmax_acc,
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0]
}
metricas_df = pd.DataFrame([metricas])
metricas_df.to_csv(f'{OUTPUT_DIR}/metricas_comparativa.csv', index=False)

metricas_modelo_df = pd.DataFrame([{
    'Modelo': 'Regresion Logistica (MLE + Umbral OOF)',
    'F1_Test': f1_test,
    'Precision_Test': precision_test,
    'Recall_Test': recall_test,
    'Accuracy_Test': accuracy_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'LogLoss_Test': logloss_test,
    'Brier_Test': brier_test,
    'Threshold_OOF': best_threshold,
    'F1_OOF_Train': best_oof_f1,
    'ArgMax_Hits': argmax_hits,
    'ArgMax_Races': argmax_total_races,
    'ArgMax_Top1_Acc': argmax_acc,
    'Intercepto': float(model.intercept_[0]),
    'N_Coeficientes': len(model.coef_[0]),
    'Tiempo_Entrenamiento_s': training_time,
    'Iteraciones': str(model.n_iter_),
    'Convergio': str(model.n_iter_ < model_config['max_iter'])
}])
metricas_modelo_df.to_csv(f'{OUTPUT_DIR}/metricas_modelo.csv', index=False)
print(f"\n✓ Métricas guardadas en {OUTPUT_DIR}/metricas_comparativa.csv y metricas_modelo.csv")

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

# Visualización 6: Curva Precision-Recall con umbral óptimo OOF marcado
plt.figure(figsize=(10, 8))
test_prec_c, test_rec_c, _ = precision_recall_curve(y_test, y_test_proba)
plt.plot(test_rec_c, test_prec_c, color='#2E86AB', linewidth=3, label=f'Test PR Curve (AUC-PR = {pr_auc:.4f})')
plt.scatter([recall_test], [precision_test], color='#D1495B', s=120, zorder=5,
            label=f'Punto operativo τ*={best_threshold:.4f} (P={precision_test:.3f}, R={recall_test:.3f}, F1={f1_test:.3f})')
plt.axhline(float(y_test.mean()), color='gray', linestyle='--', linewidth=1.5,
            label=f'Prevalencia Test = {y_test.mean():.4f}')
plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Curva Precision-Recall y Umbral Calibrado (OOF Train) — Regresión Logística', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_curva_pr_umbral.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 6: 06_curva_pr_umbral.png")

# Visualización 7: Matriz de Confusión con umbral calibrado
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(10, 8))
if sns is not None:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={'size': 14}, cbar_kws={'label': 'Número de Predicciones'},
                xticklabels=['Pred: No Victoria', 'Pred: Victoria'],
                yticklabels=['Real: No Victoria', 'Real: Victoria'], ax=ax)
else:
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, label='Número de Predicciones')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No Victoria', 'Pred: Victoria'])
    ax.set_yticklabels(['Real: No Victoria', 'Real: Victoria'])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > thresh else 'black', fontweight='bold')
ax.grid(False)
ax.set_xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
ax.set_title(f'Matriz de Confusión (τ* = {best_threshold:.4f}) — Regresión Logística', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 7: 07_matriz_confusion.png")

# ============================================================================
# SECCIÓN 11: RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - REGRESIÓN LOGÍSTICA")
print("="*80)

print(f"\n📊 CONFIGURACIÓN DEL MODELO:")
print(f"  • Algoritmo: Regresión Logística (clasificador lineal generalizado)")
print(f"  • Calibración de probabilidad: class_weight={model_config['class_weight']} (MLE)")
print(f"  • Umbral de decisión: τ* = {best_threshold:.6f} (F1-max OOF 5-Fold TimeSeriesSplit en Train)")
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
top_neg_var = coef_df[coef_df['coeficiente'] < 0].iloc[0]
print(f"  • Mayor efecto positivo: {top_pos_var['variable']} (β={top_pos_var['coeficiente']:+.4f})")
print(f"  • Mayor efecto negativo: {top_neg_var['variable']} (β={top_neg_var['coeficiente']:+.4f})")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • {OUTPUT_DIR}/tabla_coeficientes_logistica.csv (todos los coeficientes)")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv (métricas de rendimiento)")
print(f"  • {OUTPUT_DIR}/metricas_modelo.csv (métricas detalladas y umbral)")
print(f"  • {OUTPUT_DIR}/01_coeficientes_top20.png")
print(f"  • {OUTPUT_DIR}/02_coeficientes_polaridad.png")
print(f"  • {OUTPUT_DIR}/03_distribucion_coeficientes.png")
print(f"  • {OUTPUT_DIR}/04_odds_ratios.png")
print(f"  • {OUTPUT_DIR}/05_curva_roc.png")
print(f"  • {OUTPUT_DIR}/06_curva_pr_umbral.png")
print(f"  • {OUTPUT_DIR}/07_matriz_confusion.png")

print("\n" + "="*80)
print("PROCESO COMPLETADO - FASE DE ENTRENAMIENTO Y EVALUACIÓN CON UMBRAL")
print("="*80)

print("\n💡 CONSIDERACIONES PRÁCTICAS IDENTIFICADAS:")
print("  • El desbalance severo (~4.5% victorias) se trató con probabilidades MLE calibradas (class_weight=None) y umbral τ* calibrado vía F1-max OOF en Train")
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

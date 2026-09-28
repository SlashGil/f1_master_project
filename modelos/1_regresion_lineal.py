"""
========================================
MODELO 1: REGRESIÓN LINEAL
========================================

Este script implementa un modelo de regresión lineal para predecir la probabilidad
de victoria en carreras de Fórmula 1.

OBJETIVO:
  - Entrenar modelo de regresión lineal
  - Analizar importancia de variables mediante coeficientes
  - Calcular métricas de rendimiento estandarizadas
  - Generar visualizaciones comparables con otros modelos

MÉTRICAS:
  - F1-Score (para clase victoria)
  - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
  - Precision-Recall AUC
  - Precisión y Recall
  - Classification Report completo

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score,
    mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]

# Directorios
DATA_DIR = "../data"
OUTPUT_DIR = "../resultados_regresion_lineal"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MODELO 1: REGRESIÓN LINEAL - Análisis de Importancia de Variables")
print("="*80)

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================================================

print("\n[1/7] Cargando dataset procesado...")

try:
    df = pd.read_csv(f'{DATA_DIR}/dataset_tesis_f1.csv')
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  • Variables predictoras: {df.shape[1] - 1}")
    print(f"  • Variable objetivo: win")
    
    # Verificar distribución de clases
    n_victorias = df['win'].sum()
    pct_victorias = (n_victorias / len(df)) * 100
    print(f"  • Victorias: {n_victorias} ({pct_victorias:.2f}%)")
    print(f"  • No victorias: {len(df) - n_victorias} ({100-pct_victorias:.2f}%)")
    
except FileNotFoundError:
    print("✗ Error: No se encontró dataset_tesis_f1.csv")
    print("  Asegúrese de ejecutar el script de preparación de datos primero")
    exit(1)

# ============================================================================
# SECCIÓN 2: PREPARACIÓN DE DATOS
# ============================================================================

print("\n[2/7] Preparando Features y Target...")

# Separar features y target (excluyendo raceId como predictor para evitar fuga de información)
X = df.drop(columns=['win', 'raceId'], errors='ignore')
y = df['win']

print(f"Features seleccionados: {X.shape[1]}")
print(f"Target: win (binario)")

# ============================================================================
# SECCIÓN 3: DIVISIÓN DE DATOS
# ============================================================================

print("\n[3/7] Dividiendo datos con corte temporal (sin aleatoriedad)...")

if {'year', 'round', 'raceId'}.issubset(df.columns):
    df_sorted = df.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)
else:
    df_sorted = df.sort_values('raceId').reset_index(drop=True)

split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

X_train = train_df.drop(columns=['win', 'raceId'], errors='ignore')
y_train = train_df['win']
X_test = test_df.drop(columns=['win', 'raceId'], errors='ignore')
y_test = test_df['win']

print(f"✓ Entrenamiento (pasado): {X_train.shape[0]} muestras")
print(f"✓ Prueba (futuro): {X_test.shape[0]} muestras")
print("✓ Split temporal aplicado (80/20 cronológico)")

# Verificar balance en train/test
print(f"\n  Entrenamiento - Victorias: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Prueba - Victorias: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# SECCIÓN 4: ESCALADO DE CARACTERÍSTICAS
# ============================================================================

print("\n[4/7] Aplicando StandardScaler después del split temporal...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos escalados a media=0, desviación=1")
print(f"  X_train scaled: {X_train_scaled.shape}")
print(f"  X_test scaled: {X_test_scaled.shape}")

# ============================================================================
# SECCIÓN 5: ENTRENAMIENTO DE MODELO
# ============================================================================

print("\n[5/7] Entrenando modelo de Regresión Lineal...")
print("-" * 80)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✓ Modelo entrenado exitosamente")
print(f"  • Coeficientes aprendidos: {len(model.coef_)}")
print(f"  • Interceptor (bias): {model.intercept_:.4f}")

# ============================================================================
# SECCIÓN 6: PREDICCIONES
# ============================================================================

print("\n[6/7] Realizando predicciones...")

# Predicciones continuous (regression)
y_train_pred_continuous = model.predict(X_train_scaled)
y_test_pred_continuous = model.predict(X_test_scaled)

# Convertir predicciones a binarias (0/1) usando threshold 0.5
y_train_pred_binary = (y_train_pred_continuous >= 0.5).astype(int)
y_test_pred_binary = (y_test_pred_continuous >= 0.5).astype(int)

print("✓ Predicciones generadas (continuas y binarias)")

# ============================================================================
# SECCIÓN 7: EVALUACIÓN DE MÉTRICAS
# ============================================================================

print("\n" + "="*80)
print("EVALUACIÓN DE MÉTRICAS")
print("="*80)

# Métricas de regresión
train_mse = mean_squared_error(y_train, y_train_pred_continuous)
test_mse = mean_squared_error(y_test, y_test_pred_continuous)
train_r2 = r2_score(y_train, y_train_pred_continuous)
test_r2 = r2_score(y_test, y_test_pred_continuous)

print("\n[REVELANCIA DE REGRESIÓN]")
print(f"  Entrenamiento:")
print(f"    • MSE (Error Cuadrático Medio): {train_mse:.6f}")
print(f"    • R² (Coeficiente de Determinación): {train_r2:.6f}")
print(f"  Prueba:")
print(f"    • MSE: {test_mse:.6f}")
print(f"    • R²: {test_r2:.6f}")

# Métricas de clasificación binaria
print("\n[REVELANCIA DE CLASIFICACIÓN]")
print(f"  F1-Score (clase victoria):")
f1_train = f1_score(y_train, y_train_pred_binary)
f1_test = f1_score(y_test, y_test_pred_binary)
print(f"    • Entrenamiento: {f1_train:.4f}")
print(f"    • Prueba: {f1_test:.4f}")

print(f"\n  AUC-ROC (Área Bajo la Curva ROC):")
roc_auc = roc_auc_score(y_test, y_test_pred_continuous)
print(f"    • Prueba: {roc_auc:.4f}")

print(f"\n  Precision-Recall AUC:")
pr_auc = average_precision_score(y_test, y_test_pred_continuous)
print(f"    • Prueba: {pr_auc:.4f}")

print(f"\n  Métricas detalladas (Prueba):")
test_precision = precision_score(y_test, y_test_pred_binary)
test_recall = recall_score(y_test, y_test_pred_binary)
print(f"    • Precisión: {test_precision:.4f}")
print(f"    • Recall: {test_recall:.4f}")

print("\n[CLASSIFICATION REPORT COMPLETO]")
print("=" * 80)
report = classification_report(y_test, y_test_pred_binary, 
                              target_names=['No Victoria', 'Victoria'],
                              digits=4)
print(report)

# Guardar métricas en archivo CSV
metricas = {
    'Modelo': 'Regresion Lineal',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'Precision_Test': test_precision,
    'Recall_Test': test_recall,
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0]
}
metricas_df = pd.DataFrame([metricas])
metricas_df.to_csv(f'{OUTPUT_DIR}/metricas_comparativa.csv', index=False)
print(f"\n✓ Métricas guardadas en {OUTPUT_DIR}/metricas_comparativa.csv")

# ============================================================================
# SECCIÓN 8: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
print("="*80)

# Obtener coeficientes y sus magnitudes
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 20 variables más importantes (por magnitud absoluta):")
print("-" * 80)
print(f"{'Pos':<5} {'Feature':<50} {'Coeficiente':>12} {'Magnitud':>10}")
print("-" * 80)

for idx, (_, row) in enumerate(coef_df.head(20).iterrows(), 1):
    feature_name = row['feature']
    coef_val = row['coefficient']
    abs_val = row['abs_coefficient']
    
    # Truncar nombre si es muy largo
    if len(feature_name) > 48:
        feature_name = feature_name[:45] + "..."
    
    coef_sign = "+" if coef_val >= 0 else ""
    print(f"{idx:<5} {feature_name:<50} {coef_sign}{coef_val:>11.4f} {abs_val:>10.4f}")

# Guardar análisis completo
coef_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
print(f"\n✓ Análisis de importancia guardado en {OUTPUT_DIR}/feature_importance.csv")

# Identificar constructores en top features
constructor_features = [f for f in coef_df.head(20)['feature'] if 'constructorId_' in f]
nationality_features = [f for f in coef_df.head(20)['feature'] if 'nationality_' in f]

if constructor_features:
    print(f"\n📊 {len(constructor_features)} constructores en top 20 más importantes:")
    for feat in constructor_features[:5]:
        const_id = feat.split('_')[1]
        print(f"  • Constructor ID: {feat}")

if nationality_features:
    print(f"\n🌍 {len(nationality_features)} nacionalidades en top 20 más importantes:")
    for feat in nationality_features[:5]:
        print(f"  • {feat}")

# ============================================================================
# SECCIÓN 9: VISUALIZACIÓN DE RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Visualización 1: Coeficientes de Regresión (Importancia de Variables)
plt.figure(figsize=(14, 10))
top_features = coef_df.head(20)

# Determinar color basado en signo del coeficiente
colors = ['#2E86AB' if c >= 0 else '#D1495B' for c in top_features['coefficient']]

plt.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors, alpha=0.7, edgecolor='BLACK')
plt.yticks(range(len(top_features)), [f[:45] + '...' if len(f) > 45 else f for f in top_features['feature']])
plt.xlabel('|Coeficiente| (Importancia Absoluta)', fontsize=12, fontweight='bold')
plt.title('Top 20 Variables Más Determinantes para Ganar en F1\n(Regresión Lineal)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Agregar leyenda
import matplotlib.patches as mpatches
pos_patch = mpatches.Patch(color='#2E86AB', label='Coeficiente Positivo (Aumenta P(victoria))')
neg_patch = mpatches.Patch(color='#D1495B', label='Coeficiente Negativo (Reduce P(victoria))')
plt.legend(handles=[pos_patch, neg_patch], loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_regresion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 1: feature_importance_regresion.png")

# Visualización 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_continuous)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#D1495B', lw=3, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Regresión Lineal\n(Clase: Victoria de Piloto)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve_regresion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 2: roc_curve_regresion.png")

# Visualización 3: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_pred_continuous)

plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color='#2E86AB', lw=3, label=f'Curva PR (AUC = {pr_auc:.4f})')
plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
plt.ylabel('Precisión (Valor Predictivo Positivo)', fontsize=12, fontweight='bold')
plt.title('Curva Precision-Recall - Regresión Lineal\n(Importante para datos desbalanceados)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_regresion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 3: precision_recall_regresion.png")

# Visualización 4: Matriz de Confusión
cm = confusion_matrix(y_test, y_test_pred_binary)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples_r', 
           annot_kws={'size': 14}, cbar_kws={'label': 'Número de Predicciones'},
           xticklabels=['Pred: No Victoria', 'Pred: Victoria'],
           yticklabels=['Real: No Victoria', 'Real: Victoria'])
plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión - Regresión Lineal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_regresion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 4: confusion_matrix_regresion.png")

# Visualización 5: Predicciones vs Valores Reales
plt.figure(figsize=(12, 8))

# Muestra aleatoria de 2000 puntos para visualización
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)

plt.scatter(y_test_pred_continuous[sample_indices], y_test.iloc[sample_indices], 
           alpha=0.3, s=30, color='#2E86AB', edgecolors='black', linewidth=0.5)
# Línea de identidad
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Línea de Identidad (Predicción = Real)')
# Línea de decisión (threshold 0.5)
plt.axhline(y=0.5, color='orange', linestyle=':', lw=2, label='Umbral de Decisión (0.5)')

plt.xlabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real (0=No Victoria, 1=Victoria)', fontsize=12, fontweight='bold')
plt.title('Predicciones vs Valores Reales - Regresión Lineal', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/predictions_vs_actual_regresion.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 5: predictions_vs_actual_regresion.png")

# Visualización 6: Comparación de Coeficientes (Positivos vs Negativos)
plt.figure(figsize=(12, 8))

# Filtrar top 15 positivos y top 15 negativos
top_positive = coef_df[coef_df['coefficient'] > 0].head(15)
top_negative = coef_df[coef_df['coefficient'] < 0].tail(15)

colors_pos = ['#2E86AB'] * len(top_positive)
colors_neg = ['#D1495B'] * len(top_negative)

plt.barh(range(len(top_positive)), top_positive['coefficient'], 
        color=colors_pos, alpha=0.7, label='Coeficientes Positivos')
plt.barh(range(len(top_positive), len(top_positive) + len(top_negative)), 
        top_negative['coefficient'], 
        color=colors_neg, alpha=0.7, label='Coeficientes Negativos')

plt.yticks(range(len(top_positive) + len(top_negative)), 
          list(top_positive['feature'][:15]) + list(top_negative['feature'][:15]))
plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
plt.xlabel('Valor del Coeficiente', fontsize=12, fontweight='bold')
plt.title('Coeficientes de Regresión - Top Positivos y Negativos', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/coefficients_polaridad.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 6: coefficients_polaridad.png")

# ============================================================================
# SECCIÓN 10: RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - REGRESIÓN LINEAL")
print("="*80)

print(f"\n📊 MÉTRICAS PRINCIPALES:")
print(f"  • F1-Score (clase victoria):        {f1_test:.4f}")
print(f"  • AUC-ROC:                           {roc_auc:.4f}")
print(f"  • AUC-Precision-Recall               {pr_auc:.4f}")
print(f"  • Precisión (clase victoria):         {test_precision:.4f}")
print(f"  • Recall (clase victoria):           {test_recall:.4f}")

print(f"\n🔢 MÉTRICAS DE REGRESIÓN:")
print(f"  • R² (Coeficiente de Determinación): {test_r2:.4f}")
print(f"  • MSE (Error Cuadrático Medio):      {test_mse:.6f}")

print(f"\n🎯 VARIABLE MÁS IMPORTANTE:")
top_feature = coef_df.iloc[0]
print(f"  • Nombre:  {top_feature['feature']}")
print(f"  • Coefiente: {top_feature['coefficient']:.4f}")
print(f"  • Interpretación:", 
      'Aumenta significativamente P(victoria)' if top_feature['coefficient'] > 0 
      else 'Reduce significativamente P(victoria)')

print(f"\n📈 INTERPRETACIÓN DE RESULTADOS:")
if test_r2 > 0.3:
    print(f"  • El modelo explica {(test_r2*100):.1f}% de la varianza en 'win'")
    print(f"  • Regresión lineal captura relaciones predictivas moderadamente fuertes")
else:
    print(f"  • El modelo explica solo {(test_r2*100):.1f}% de la varianza")
    print(f"  • Sugerencia: Considerar modelos no-lineales (Random Forest, MLP)")

if f1_test < 0.3:
    print(f"  • F1-Score bajo ({f1_test:.4f}) indica dificultad prediciendo victorias")
    print(f"  • Posible causa: Dataset desbalanceado (solo ~5% victorias)")
    print(f"  • Sugerencia: Implementar técnicas de balanceo de clases")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv")
print(f"  • {OUTPUT_DIR}/feature_importance.csv")
print(f"  • {OUTPUT_DIR}/feature_importance_regresion.png")
print(f"  • {OUTPUT_DIR}/roc_curve_regresion.png")
print(f"  • {OUTPUT_DIR}/precision_recall_regresion.png")
print(f"  • {OUTPUT_DIR}/confusion_matrix_regresion.png")
print(f"  • {OUTPUT_DIR}/predictions_vs_actual_regresion.png")
print(f"  • {OUTPUT_DIR}/coefficients_polaridad.png")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

print("\n🎉 Modelo de Regresión Lineal entrenado y evaluado exitosamente")
print("📊 Visualizaciones y análisis de importancia generados")
print("📁 Resultados guardados en:", OUTPUT_DIR)

print("\n📋 PRÓXIMOS PASOS:")
print("  1. Comparar resultados con Random Forest y Perceptrón Multicapa")
print("  2. Analizar convergencia de importancia de variables entre modelos")
print("  3. Documentar patrones consistentes determinantes para victoria")
print("  4. Interpretar coeficientes en contexto del automovilismo profesional")

print("\n💡 SUGERENCIAS PARA ANÁLISIS:")
print("  • Revisar coeficientes de constructores para identificar escuderías dominantes")
print("  • Analizar variables de grid y fastestLapSpeed como predictores de ventaja")
print("  • Considerar añadir interacción entre constructorId y nationality (si no hay colinealidad severa)")
print("  • Evaluar necesidad de regularización (Ridge/Lasso) para reducir overfitting")

print("\n✨ LISTO PARA COMPARACIÓN CON OTROS MODELOS ✨")
print("   Los archivos .csv y .png permitirán comparación sistemática")
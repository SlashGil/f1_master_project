"""
========================================
MODELO 2: RANDOM FOREST
========================================

Este script implementa un modelo Random Forest para predecir la probabilidad
de victoria en carreras de Fórmula 1.

OBJETIVO:
  - Entrenar modelo Random Forest con balance de clases
  - Analizar importancia de variables mediante feature importance nativo
  - Calcular métricas de rendimiento estandarizadas
  - Generar visualizaciones comparables con otros modelos

MÉTRICAS:
  - F1-Score (para clase victoria)
  - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
  - Precision-Recall AUC
  - Precisión y Recall
  - Classification Report completo

PARÁMETROS DE ENTRENAMIENTO:
  - n_estimators=100: 100 árboles en el bosque
  - max_depth=10: Limita profundidad para evitar overfitting
  - min_samples_split=5: Mínimo de muestras para dividir nodo
  - class_weight='balanced': Manejo de dataset desbalanceado

AUTOR: Proyecto Tesis F1
FECHA: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score
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
OUTPUT_DIR = "../resultados_random_forest"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MODELO 2: RANDOM FOREST - Análisis de Importancia de Variables")
print("="*80)

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================================================

print("\n[1/8] Cargando dataset procesado...")

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

print("\n[2/8] Preparando Features y Target...")

# Separar features y target
X = df.drop('win', axis=1)
y = df['win']

print(f"Features seleccionados: {X.shape[1]}")
print(f"Target: win (binario)")

# ============================================================================
# SECCIÓN 3: DIVISIÓN DE DATOS
# ============================================================================

print("\n[3/8] Dividiendo datos en entrenamiento y prueba...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Estratificación para mantener proporción de clases
)

print(f"✓ Entrenamiento: {X_train.shape[0]} muestras")
print(f"✓ Prueba: {X_test.shape[0]} muestras")
print(f"✓ Estratificación aplicada para mantener distribución de clases")

# Verificar balance en train/test
print(f"\n  Entrenamiento - Victorias: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Prueba - Victorias: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ============================================================================
# SECCIÓN 4: ESCALADO DE CARACTERÍSTICAS
# ============================================================================

print("\n[4/8] Aplicando StandardScaler a características...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos escalados a media=0, desviación=1")
print(f"  X_train scaled: {X_train_scaled.shape}")
print(f"  X_test scaled: {X_test_scaled.shape}")

# ============================================================================
# SECCIÓN 5: ENTRENAMIENTO DE MODELO RANDOM FOREST
# ============================================================================

print("\n[5/8] Entrenando modelo Random Forest...")
print("-" * 80)

print("Configuración de hiperparámetros:")
print("  • n_estimators = 100 (número de árboles)")
print("  • max_depth = 10 (profundidad máxima)")
print("  • min_samples_split = 5")
print("  • class_weight = 'balanced' (manejo de desbalance)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Utilizar todos los núcleos disponibles
)

model.fit(X_train, y_train)

print("✓ Modelo Random Forest entrenado exitosamente")
print(f"  • Número de árboles entrenados: {len(model.estimators_)}")

# ============================================================================
# SECCIÓN 6: PREDICCIONES
# ============================================================================

print("\n[6/8] Realizando predicciones...")

# Predicciones de probabilidad
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Predicciones binarias (0/1)
y_train_pred_binary = model.predict(X_train)
y_test_pred_binary = model.predict(X_test)

print("✓ Predicciones generadas (probabilidades y binarias)")

# ============================================================================
# SECCIÓN 7: EVALUACIÓN DE MÉTRICAS
# ============================================================================

print("\n" + "="*80)
print("EVALUACIÓN DE MÉTRICAS")
print("="*80)

# Métricas de clasificación binaria
print("\n[REVELANCIA DE CLASIFICACIÓN]")
print(f"  F1-Score (clase victoria):")
f1_train = f1_score(y_train, y_train_pred_binary)
f1_test = f1_score(y_test, y_test_pred_binary)
print(f"    • Entrenamiento: {f1_train:.4f}")
print(f"    • Prueba: {f1_test:.4f}")

print(f"\n  AUC-ROC (Área Bajo la Curva ROC):")
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"    • Prueba: {roc_auc:.4f}")

print(f"\n  Precision-Recall AUC:")
pr_auc = average_precision_score(y_test, y_test_pred_proba)
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
    'Modelo': 'Random Forest',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'Precision_Test': test_precision,
    'Recall_Test': test_recall,
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0],
    'N_Estimators': 100,
    'Max_Depth': 10
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

# Feature importance nativo de Random Forest
importances = model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 20 variables más importantes (Feature Importance):")
print("-" * 80)
print(f"{'Pos':<5} {'Feature':<50} {'Importancia':>10}")
print("-" * 80)

for idx, (_, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
    feature_name = row['feature']
    importance = row['importance']
    
    # Truncar nombre si es muy largo
    if len(feature_name) > 46:
        feature_name = feature_name[:43] + "..."
    
    print(f"{idx:<5} {feature_name:<50} {importance:>10.4f}")

# Guardar análisis completo
feature_importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
print(f"\n✓ Análisis de importancia guardado en {OUTPUT_DIR}/feature_importance.csv")

# Estadísticas de importancia
print(f"\nEstadísticas de importancia de variables:")
print(f"  • Importancia máxima: {feature_importance_df['importance'].max():.6f}")
print(f"  • Importancia mínima: {feature_importance_df['importance'].min():.6f}")
print(f"  • Importancia media: {feature_importance_df['importance'].mean():.6f}")
print(f"  • Desviación estándar: {feature_importance_df['importance'].std():.6f}")

# Identificar constructores en top features
constructor_features = [f for f in feature_importance_df.head(20)['feature'] if 'constructorId_' in f]
nationality_features = [f for f in feature_importance_df.head(20)['feature'] if 'nationality_' in f]
numeric_features = [f for f in feature_importance_df.head(20)['feature'] if '_' not in f]

print(f"\n📊 Distribución de tipos de variables en top 20:")
print(f"  • Nacionalidades: {len(nationality_features)}")
print(f"  • Constructores: {len(constructor_features)}")
print(f"  • Variables numéricas: {len(numeric_features)}")

# ============================================================================
# SECCIÓN 9: VISUALIZACIÓN DE RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("GENERANDO VISUALIZACIONES")
print("="*80)

# Visualización 1: Feature Importance
plt.figure(figsize=(14, 10))
top_features = feature_importance_df.head(20)

# Paleta de colores gradientes
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))

plt.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black')
plt.yticks(range(len(top_features)), [f[:45] + '...' if len(f) > 45 else f for f in top_features['feature']])
plt.xlabel('Importance (Mean Decrease in Impurity)', fontsize=12, fontweight='bold')
plt.title('Top 20 Variables Más Determinantes para Ganar en F1\n(Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 1: feature_importance_rf.png")

# Visualización 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#D1495B', lw=3, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Random Forest\n(Clase: Victoria de Piloto)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 2: roc_curve_rf.png")

# Visualización 3: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color='#2E86AB', lw=3, label=f'Curva PR (AUC = {pr_auc:.4f})')
plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
plt.ylabel('Precisión (Valor Predictivo Positivo)', fontsize=12, fontweight='bold')
plt.title('Curva Precision-Recall - Random Forest\n(Importante para datos desbalanceados)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 3: precision_recall_rf.png")

# Visualización 4: Matriz de Confusión
cm = confusion_matrix(y_test, y_test_pred_binary)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges_r', 
           annot_kws={'size': 14}, cbar_kws={'label': 'Número de Predicciones'},
           xticklabels=['Pred: No Victoria', 'Pred: Victoria'],
           yticklabels=['Real: No Victoria', 'Real: Victoria'])
plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 4: confusion_matrix_rf.png")

# Visualización 5: Predicciones vs Valores Reales
plt.figure(figsize=(12, 8))

# Muestra aleatoria de 2000 puntos para visualización
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)

plt.scatter(y_test_pred_proba[sample_indices], y_test.iloc[sample_indices], 
           alpha=0.4, s=25, color='#2E86AB', edgecolors='black', linewidth=0.3)
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Línea de Identidad (Predicción = Real)')
plt.axhline(y=0.5, color='orange', linestyle=':', lw=2, label='Umbral de Decisión (0.5)')

plt.xlabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real (0=No Victoria, 1=Victoria)', fontsize=12, fontweight='bold')
plt.title('Predicciones vs Valores Reales - Random Forest', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/predictions_vs_actual_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 5: predictions_vs_actual_rf.png")

# Visualización 6: Distribución de Importancia de Variables
plt.figure(figsize=(12, 6))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma de importancias
ax1.hist(feature_importance_df['importance'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
ax1.axvline(feature_importance_df['importance'].median(), color='red', linestyle='--', linewidth=2, label=f'Mediana: {feature_importance_df["importance"].median():.6f}')
ax1.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax1.set_title('Distribución de Importancia de Variables\n(Todas las Features)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Boxplot de importancias
bp = ax2.boxplot(feature_importance_df['importance'], patch_artist=True)
bp['boxes'][0].set_facecolor('#D1495B')
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Importance', fontsize=11, fontweight='bold')
ax2.set_title('Boxplot de Importancia de Variables', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/importance_distribution_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 6: importance_distribution_rf.png")

# Visualización 7: Importancia por Tipo de Variable
plt.figure(figsize=(10, 6))

# Agrupar importancias por tipo
numeric_importance = feature_importance_df[feature_importance_df['feature'].apply(lambda x: '_' not in x)]['importance'].sum()
constructor_importance = feature_importance_df[feature_importance_df['feature'].str.contains('constructorId_')]['importance'].sum()
nationality_importance = feature_importance_df[feature_importance_df['feature'].str.contains('nationality_')]['importance'].sum()

tipos = ['Variables Numéricas', 'ConstructorID', 'Nationality']
importancias = [numeric_importance, constructor_importance, nationality_importance]
colores = ['#2E86AB', '#D1495B', '#F0E68C']

bars = plt.bar(tipos, importancias, color=colores, alpha=0.7, edgecolor='black')
plt.xlabel('Tipo de Variable', fontsize=12, fontweight='bold')
plt.ylabel('Importancia Total Sumada', fontsize=12, fontweight='bold')
plt.title('Importancia Acumulada por Tipo de Variable\n(Random Forest)', fontsize=14, fontweight='bold')

# Agregar etiquetas
for bar, imp in zip(bars, importancias):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/importance_by_type_rf.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 7: importance_by_type_rf.png")

# ============================================================================
# SECCIÓN 10: RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - RANDOM FOREST")
print("="*80)

print(f"\n📊 MÉTRICAS PRINCIPALES:")
print(f"  • F1-Score (clase victoria):        {f1_test:.4f}")
print(f"  • AUC-ROC:                           {roc_auc:.4f}")
print(f"  • AUC-Precision-Recall               {pr_auc:.4f}")
print(f"  • Precisión (clase victoria):         {test_precision:.4f}")
print(f"  • Recall (clase victoria):           {test_recall:.4f}")

print(f"\n🎯 VARIABLE MÁS IMPORTANTE:")
top_feature = feature_importance_df.iloc[0]
print(f"  • Nombre:  {top_feature['feature']}")
print(f"  • Feature Importance: {top_feature['importance']:.6f}")
print(f"  • Contribución relativa: {(top_feature['importance'] / feature_importance_df['importance'].sum() * 100):.2f}% del total")

print(f"\n📈 INTERPRETACIÓN DE RESULTADOS:")
print(f"  • {len(model.estimators_)} árboles entrenados para robustez del modelo")
print(f"  • Profundidad limitada (max_depth=10) para prevenir overfitting")
print(f"  • class_weight='balanced' maneja dataset desbalanceado")

if f1_test > 0.5:
    print(f"  • F1-Score moderado-alto ({f1_test:.4f}) indica buen rendimiento")
else:
    print(f"  • F1-Score moderado ({f1_test:.4f}) puede deberse a datos desbalanceados")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv")
print(f"  • {OUTPUT_DIR}/feature_importance.csv")
print(f"  • {OUTPUT_DIR}/feature_importance_rf.png")
print(f"  • {OUTPUT_DIR}/roc_curve_rf.png")
print(f"  • {OUTPUT_DIR}/precision_recall_rf.png")
print(f"  • {OUTPUT_DIR}/confusion_matrix_rf.png")
print(f"  • {OUTPUT_DIR}/predictions_vs_actual_rf.png")
print(f"  • {OUTPUT_DIR}/importance_distribution_rf.png")
print(f"  • {OUTPUT_DIR}/importance_by_type_rf.png")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

print("\n🎉 Modelo Random Forest entrenado y evaluado exitosamente")
print("📊 Visualizaciones y análisis de importancia generados")
print("📁 Resultados guardados en:", OUTPUT_DIR)

print("\n📋 PRÓXIMOS PASOS:")
print("  1. Comparar resultados con Regresión Lineal y Perceptrón Multicapa")
print("  2. Analizar convergencia de importancia de variables entre modelos")
print("  3. Documentar patrones consistentes determinantes para victoria")
print("  4. Evaluar si Random Forest captura relaciones no lineales mejor que LR")

print("\n💡 VENTAJAS DE RANDOM FOREST:")
print("  • Captura relaciones no lineales e interacciones entre variables")
print("  • Robusto al overfitting (ensamble de múltiples árboles)")
print("  • Maneja bien datos desbalanceados (class_weight)")
print("  • Feature importance nativo interpretativo")

print("\n✨ LISTO PARA COMPARACIÓN CON OTROS MODELOS ✨")
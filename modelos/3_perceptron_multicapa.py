"""
========================================
MODELO 3: PERCEPTRÓN MULTICAPA (MLP)
========================================

Este script implementa un Perceptrón Multicapa (Red Neuronal) para predecir
la probabilidad de victoria en carreras de Fórmula 1.

OBJETIVO:
  - Entrenar modelo de red neuronal profunda con dropout y early stopping
  - Analizar importancia de variables mediante análisis de pesos de la primera capa
  - Calcular métricas de rendimiento estandarizadas
  - Generar visualizaciones comparables con otros modelos

MÉTRICAS:
  - F1-Score (para clase victoria)
  - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
  - Precision-Recall AUC
  - Precisión y Recall
  - Classification Report completo

ARQUITECTURA DE RED:
  - Capa de entrada: Dimensionalidad numérica del dataset
  - Capa oculta 1: 64 neuronas, activación ReLU, Dropout 0.5
  - Capa oculta 2: 32 neuronas, activación ReLU, Dropout 0.5
  - Capa de salida: 1 neurona, activación Sigmoid (probabilidad binaria)

OPTIMIZACIÓN:
  - Optimizador: Adam (convergencia rápida)
  - Pérdida: Binary cross-entropy (apropiado para clasificación binaria)
  - Regularización: Dropout 0.5 (previene overfitting)

AUTOR: Proyecto Tesis F1
FECHA: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Suprimir warnings de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]

# Directorios
DATA_DIR = "../data"
OUTPUT_DIR = "../resultados_mlp"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MODELO 3: PERCEPTRÓN MULTICAPA (MLP) - Análisis de Importancia de Variables")
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
print("\n⚠️ IMPORTANT: MLP requiere escalado debido a:")
print("   • Optimización del gradiente converge más rápido")
print("   • Evita saturación de funciones de activación")

# ============================================================================
# SECCIÓN 5: CONSTRUCCIÓN DEL MODELO MLR (MLP)
# ============================================================================

print("\n[5/8] Construyendo arquitectura de Red Neuronal...")
print("-" * 80)

print("Arquitectura:")
print("  • Capa de entrada: {} neuronas (una por feature)".format(X.shape[1]))
print("  • Capa oculta 1: 64 neuronas, ReLU, Dropout 0.5")
print("  • Capa oculta 2: 32 neuronas, ReLU, Dropout 0.5")
print("  • Capa de salida: 1 neurona, Sigmoid (probabilidad binaria)")

model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu', name='dense_1'),
    Dropout(0.5, name='dropout_1'),
    Dense(32, activation='relu', name='dense_2'),
    Dropout(0.5, name='dropout_2'),
    Dense(1, activation='sigmoid', name='output')
])

# Compilación del modelo
print("\nCompilación:")
print("  • Optimizador: Adam")
print("  • Función de pérdida: Binary cross-entropy (clase victoria)")
print("  • Métricas: Accuracy")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modelo compilado exitosamente")
model.summary()

# ============================================================================
# SECCIÓN 6: ENTRENAMIENTO CON EARLY STOPPING
# ============================================================================

print("\n[6/8] Entrenando modelo con Early Stopping...")
print("-" * 80)

# Early stopping para prevenir overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Entrenamiento
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\n✓ Entrenamiento completado")
print(f"  • Epochs ejecutadas: {len(history.history['loss'])}")
print(f"  • Early stopping activó en época:", len(history.history['loss']))

# ============================================================================
# SECCIÓN 7: PREDICCIONES
# ============================================================================

print("\n[7/8] Realizando predicciones...")

# Predicciones de probabilidad
y_train_pred_proba = model.predict(X_train_scaled, verbose=0).flatten()
y_test_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()

# Predicciones binarias (umbral 0.5)
y_train_pred_binary = (y_train_pred_proba >= 0.5).astype(int)
y_test_pred_binary = (y_test_pred_proba >= 0.5).astype(int)

print("✓ Predicciones generadas (probabilidades y binarias)")

# ============================================================================
# SECCIÓN 8: EVALUACIÓN DE MÉTRICAS
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
    'Modelo': 'Perceptron Multicapa (MLP)',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'Precision_Test': test_precision,
    'Recall_Test': test_recall,
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0],
    'Epochs_Ejecutadas': len(history.history['loss'])
}
metricas_df = pd.DataFrame([metricas])
metricas_df.to_csv(f'{OUTPUT_DIR}/metricas_comparativa.csv', index=False)
print(f"\n✓ Métricas guardadas en {OUTPUT_DIR}/metricas_comparativa.csv")

# ============================================================================
# SECCIÓN 9: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE IMPORTANCIA DE VARIABLES")
print("="*80)

# Extraer pesos de la primera capa densa
weights = model.get_layer('dense_1').get_weights()[0]  # shape: (input_dim, 64)
biases = model.get_layer('dense_1').get_weights()[1]  # shape: (64,)

print("\nAnálisis de pesos de la primera capa (Dense 1):")
print(f"  • Forma de matriz de pesos: {weights.shape}")
print(f"  • Dimensión de entrada: {weights.shape[0]} (número de features)")
print(f"  • Número de neuronas en primera capa: {weights.shape[1]}")

# Calcular importancia como media absoluta de pesos por feature
importance = np.mean(np.abs(weights), axis=1)

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 20 variables más importantes (por peso absoluto en primera capa):")
print("-" * 80)
print(f"{'Pos':<5} {'Feature':<50} {'Importancia':>10}")
print("-" * 80)

for idx, (_, row) in enumerate(feature_importance_df.head(20).iterrows(), 1):
    feature_name = row['feature']
    importance_val = row['importance']
    
    # Truncar nombre si es muy largo
    if len(feature_name) > 46:
        feature_name = feature_name[:43] + "..."
    
    print(f"{idx:<5} {feature_name:<50} {importance_val:>10.6f}")

# Guardar análisis completo
feature_importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
print(f"\n✓ Análisis de importancia guardado en {OUTPUT_DIR}/feature_importance.csv")

# Estadísticas de importancia
print(f"\nEstadísticas de importancia de variables:")
print(f"  • Importancia máxima: {feature_importance_df['importance'].max():.6f}")
print(f"  • Importancia mínima: {feature_importance_df['importance'].min():.6f}")
print(f"  • Importancia media: {feature_importance_df['importance'].mean():.6f}")
print(f"  • Desviación estándar: {feature_importance_df['importance'].std():.6f}")

# Identificar constructores y nacionalidades en top
constructor_features = [f for f in feature_importance_df.head(20)['feature'] if 'constructorId_' in f]
nationality_features = [f for f in feature_importance_df.head(20)['feature'] if 'nationality_' in f]
numeric_features = [f for f in feature_importance_df.head(20)['feature'] if '_' not in f]

print(f"\n📊 Distribución de tipos de variables en top 20:")
print(f"  • Nacionalidades: {len(nationality_features)}")
print(f"  • Constructores: {len(constructor_features)}")
print(f"  • Variables numéricas: {len(numeric_features)}")

# ============================================================================
# SECCIÓN 10: VISUALIZACIÓN DE RESULTADOS
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
plt.xlabel('Importancia (Media de |Pesos| Primera Capa)', fontsize=12, fontweight='bold')
plt.title('Top 20 Variables Más Determinantes para Ganar en F1\n(Perceptrón Multicapa)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 1: feature_importance_mlp.png")

# Visualización 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#D1495B', lw=3, label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12, fontweight='bold')
plt.title('Curva ROC - Perceptrón Multicapa\n(Clase: Victoria de Piloto)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curve_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 2: roc_curve_mlp.png")

# Visualización 3: Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color='#2E86AB', lw=3, label=f'Curva PR (AUC = {pr_auc:.4f})')
plt.xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
plt.ylabel('Precisión (Valor Predictivo Positivo)', fontsize=12, fontweight='bold')
plt.title('Curva Precision-Recall - Perceptrón Multicapa\n(Importante para datos desbalanceados)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 3: precision_recall_mlp.png")

# Visualización 4: Matriz de Confusión
cm = confusion_matrix(y_test, y_test_pred_binary)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples_r', 
           annot_kws={'size': 14}, cbar_kws={'label': 'Número de Predicciones'},
           xticklabels=['Pred: No Victoria', 'Pred: Victoria'],
           yticklabels=['Real: No Victoria', 'Real: Victoria'])
plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
plt.title('Matriz de Confusión - Perceptrón Multicapa', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 4: confusion_matrix_mlp.png")

# Visualización 5: Curvas de Entrenamiento
plt.figure(figsize=(14, 6))

plt.plot(history.history['loss'], label='Train Loss', color='#D1495B', lw=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='#2E86AB', lw=2)
plt.axvline(x=len(history.history['loss']) - 1, color='gray', linestyle=':', 
           lw=2, label=f'Early Stop (Epoch {len(history.history["loss"])})')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
plt.title('Curvas de Entrenamiento y Validación\n(Perceptrón Multicapa)', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_history_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 5: training_history_mlp.png")

# Visualización 6: Predicciones vs Valores Reales
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
plt.title('Predicciones vs Valores Reales - Perceptrón Multicapa', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/predictions_vs_actual_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 6: predictions_vs_actual_mlp.png")

# Visualización 7: Distribución de Importancia de Variables
plt.figure(figsize=(12, 6))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma de importancias
ax1.hist(feature_importance_df['importance'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
ax1.axvline(feature_importance_df['importance'].median(), color='red', linestyle='--', linewidth=2, 
           label=f'Mediana: {feature_importance_df["importance"].median():.6f}')
ax1.set_xlabel('Importancia', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax1.set_title('Distribución de Importancia de Variables\n(Todas las Features)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Boxplot de importancias
bp = ax2.boxplot(feature_importance_df['importance'], patch_artist=True)
bp['boxes'][0].set_facecolor('#D1495B')
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Importancia', fontsize=11, fontweight='bold')
ax2.set_title('Boxplot de Importancia de Variables', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/importance_distribution_mlp.png', dpi=300, bbox_inches='tight')
print("✓ Visualización 7: importance_distribution_mlp.png")

# ============================================================================
# SECCIÓN 11: RESUMEN EJECUTIVO
# ============================================================================

print("\n" + "="*80)
print("RESUMEN EJECUTIVO - PERCEPTRÓN MULTICAPA")
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
print(f"  • Feature Importance (|pesos|): {top_feature['importance']:.6f}")
print(f"  • Interpretación: Mayor peso en primera capa → mayor impacto en transformación")

print(f"\n📈 INTERPRETACIÓN DE RESULTADOS:")
print(f"  • Arquitectura: {X.shape[1]} → 64 → 32 → 1 (input → hidden1 → hidden2 → output)")
print(f"  • Epochs hasta convergencia: {len(history.history['loss'])}")
print(f"  • Early stopping previno overfitting")
print(f"  • Dropout 0.5 regularizó el modelo")

if f1_test > 0.5:
    print(f"  • F1-Score moderado-alto ({f1_test:.4f}) indica buen rendimiento")
else:
    print(f"  • F1-Score moderado ({f1_test:.4f}) puede deberse a datos desbalanceados")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv")
print(f"  • {OUTPUT_DIR}/feature_importance.csv")
print(f"  • {OUTPUT_DIR}/feature_importance_mlp.png")
print(f"  • {OUTPUT_DIR}/roc_curve_mlp.png")
print(f"  • {OUTPUT_DIR}/precision_recall_mlp.png")
print(f"  • {OUTPUT_DIR}/confusion_matrix_mlp.png")
print(f"  • {OUTPUT_DIR}/training_history_mlp.png")
print(f"  • {OUTPUT_DIR}/predictions_vs_actual_mlp.png")
print(f"  • {OUTPUT_DIR}/importance_distribution_mlp.png")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

print("\n🎉 Modelo Perceptrón Multicapa entrenado y evaluado exitosamente")
print("📊 Visualizaciones y análisis de importancia generados")
print("📁 Resultados guardados en:", OUTPUT_DIR)

print("\n📋 PRÓXIMOS PASOS:")
print("  1. Comparar resultados con Regresión Lineal y Random Forest")
print("  2. Analizar convergencia de importancia de variables entre modelos")
print("  3. Documentar patrones consistentes determinantes para victoria")
print("  4. Evaluar si MLP captura relaciones complejas mejor que LR y RF")

print("\n💡 VENTAJAS Y LIMITACIONES DE MLP:")
print("  ✓ Ventajas:")
print("    • Aprende representaciones complejas sin ingeniería manual")
print("    • Captura relaciones no lineales profundas")
print("    • Flexible para diferentes arquitecturas")
print("  ✗ Limitaciones:")
print("    • Black-box (menos interpretable que LR)")
print("    • Requiere más datos para evitar overfitting")
print("    • Sensitive a inicialización de pesos")

print("\n✨ LISTO PARA COMPARACIÓN CON OTROS MODELOS ✨")

# Guardar modelo para futuras predicciones
model.save(f'{OUTPUT_DIR}/modelo_mlp.h5')
print(f"\n💾 Modelo guardado en: {OUTPUT_DIR}/modelo_mlp.h5")
print("   (Puede cargarlo con: keras.models.load_model('modelo_mlp.h5'))")
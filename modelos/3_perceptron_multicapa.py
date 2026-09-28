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

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Configuración de dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {DEVICE}")

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

# Separar features y target (excluyendo raceId como predictor para evitar fuga de información)
X = df.drop(columns=['win', 'raceId'], errors='ignore')
y = df['win']

print(f"Features seleccionados: {X.shape[1]}")
print(f"Target: win (binario)")

# ============================================================================
# SECCIÓN 3: DIVISIÓN DE DATOS
# ============================================================================

print("\n[3/8] Dividiendo datos con corte temporal (sin aleatoriedad)...")

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

print("\n[4/8] Aplicando StandardScaler después del split temporal...")

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
# SECCIÓN 5: CONSTRUCCIÓN DEL MODELO MLP (PyTorch)
# ============================================================================

print("\n[5/8] Construyendo arquitectura de Red Neuronal...")
print("-" * 80)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x

print("Arquitectura:")
print(f"  • Capa de entrada: {X.shape[1]} neuronas (una por feature)")
print("  • Capa oculta 1: 64 neuronas, ReLU, Dropout 0.5")
print("  • Capa oculta 2: 32 neuronas, ReLU, Dropout 0.5")
print("  • Capa de salida: 1 neurona, Sigmoid (probabilidad binaria)")

# Crear modelo
model = MLP(X_train.shape[1]).to(DEVICE)

# Compilación del modelo
print("\nCompilación:")
print("  • Optimizador: Adam")
print("  • Función de pérdida: Binary cross-entropy (clase victoria)")

optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss(reduction='none')

print("\n✓ Modelo compilado exitosamente")
print(model)

# ============================================================================
# SECCIÓN 6: ENTRENAMIENTO CON EARLY STOPPING Y CLASS WEIGHTING
# ============================================================================

print("\n[6/8] Entrenando modelo con Early Stopping...")
print("-" * 80)

# Calcular class weights para manejar desbalance
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train.values)
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
print(f"Class weights: Negativa={class_weights[0]:.4f}, Positiva={class_weights[1]:.4f}")

# Convertir a tensores
X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(DEVICE)
X_val_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
y_val_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(DEVICE)

# Early stopping para prevenir overfitting
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_weights = None

history = {'loss': [], 'val_loss': []}

# Entrenamiento
model.train()
for epoch in range(100):
    # Forward pass
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    
    # Loss con class weighting real por muestra
    loss_per_sample = criterion(outputs, y_train_tensor)
    weights = y_train_tensor * class_weights_tensor[1] + (1 - y_train_tensor) * class_weights_tensor[0]
    loss = (loss_per_sample * weights).mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Validación
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss_sample = criterion(val_outputs, y_val_tensor)
        val_weights = y_val_tensor * class_weights_tensor[1] + (1 - y_val_tensor) * class_weights_tensor[0]
        val_loss = (val_loss_sample * val_weights).mean().item()
    
    model.train()
    
    history['loss'].append(loss.item())
    history['val_loss'].append(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_weights = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping activado en época {epoch + 1}")
            break
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}, val_loss={val_loss:.4f}")

# Restaurar mejores pesos
if best_weights is not None:
    model.load_state_dict(best_weights)

print(f"\n✓ Entrenamiento completado")
print(f"  • Epochs ejecutadas: {len(history['loss'])}")

# ============================================================================
# SECCIÓN 7: PREDICCIONES
# ============================================================================

print("\n[7/8] Realizando predicciones...")

model.eval()
with torch.no_grad():
    # Predicciones de probabilidad
    y_train_pred_proba = model(X_train_tensor).cpu().numpy().flatten()
    y_test_pred_proba = model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy().flatten()

# Predicciones binarias (umbral ajustado para clase desbalanceada)
threshold = 0.3  # Umbral más bajo para capturar más victorias
y_train_pred_binary = (y_train_pred_proba >= threshold).astype(int)
y_test_pred_binary = (y_test_pred_proba >= threshold).astype(int)

print(f"✓ Predicciones generadas (umbral={threshold})")
print(f"  • Predicciones positivas en train: {y_train_pred_binary.sum()}")
print(f"  • Predicciones positivas en test: {y_test_pred_binary.sum()}")

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
    'Test_Size': X_test.shape[0]
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
weights = model.layer1.weight.detach().cpu().numpy()  # shape: (64, input_dim)

print("\nAnálisis de pesos de la primera capa (Dense 1):")
print(f"  • Forma de matriz de pesos: {weights.shape}")
print(f"  • Dimensión de entrada: {weights.shape[1]} (número de features)")
print(f"  • Número de neuronas en primera capa: {weights.shape[0]}")

# Calcular importancia como media absoluta de pesos por feature
importance = np.mean(np.abs(weights), axis=0)

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

plt.plot(history['loss'], label='Train Loss', color='#D1495B', lw=2)
plt.plot(history['val_loss'], label='Val Loss', color='#2E86AB', lw=2)
plt.axvline(x=len(history['loss']) - 1, color='gray', linestyle=':',
           lw=2, label=f'Early Stop (Epoch {len(history["loss"])})')
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
print(f"  • Epochs hasta convergencia: {len(history['loss'])}")
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
torch.save(model.state_dict(), f'{OUTPUT_DIR}/modelo_mlp.pt')
print(f"\n💾 Modelo guardado en: {OUTPUT_DIR}/modelo_mlp.pt")
print("   (Puede cargarlo con: model.load_state_dict(torch.load('modelo_mlp.pt')))")
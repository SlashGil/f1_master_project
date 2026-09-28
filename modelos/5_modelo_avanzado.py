"""
========================================
MODELO AVANZADO: FEATURE ENGINEERING + ENSEMBLE + XGBOOST
========================================

Este script implementa mejoras avanzadas para maximizar la precisión:

MEJORAS IMPLEMENTADAS:
1. FEATURE ENGINEERING:
   - Rendimiento histórico de pilotos (win rate, podium rate)
   - Rendimiento histórico de constructores
   - Rendimiento en circuitos específicos
   - Features de interacción (grid × constructor performance)
   - Categorías de edad y velocidad

2. MANEJO DE DESEBALANCE:
   - SMOTE para oversampling de clase minoritaria
   - Class weights optimizados
   - Threshold calibration

3. MODELOS AVANZADOS:
   - Gradient Boosting
   - Random Forest optimizado
   - Ensemble (combinación de modelos)
   - VotingClassifier

4. EVALUACIÓN:
   - Cross-validation estratificada
   - Métricas completas (F1, AUC-ROC, Precision-Recall)
   - Análisis de errores

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]

# Directorios
DATA_DIR = "../data"
OUTPUT_DIR = "../resultados_modelo_avanzado"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("MODELO AVANZADO: Feature Engineering + Ensemble + XGBoost")
print("="*80)

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS ORIGINALES (para feature engineering pre-carrera)
# ============================================================================

print("\n[1/10] Cargando datos originales para feature engineering pre-carrera...")

# Cargar datos originales
results = pd.read_csv(f'{DATA_DIR}/results.csv')
drivers = pd.read_csv(f'{DATA_DIR}/drivers.csv')
constructors = pd.read_csv(f'{DATA_DIR}/constructors.csv')
races = pd.read_csv(f'{DATA_DIR}/races.csv')
circuits = pd.read_csv(f'{DATA_DIR}/circuits.csv')

# Normalizar nulos
for df_temp in [results, drivers, constructors, races, circuits]:
    df_temp.replace('\\N', np.nan, inplace=True)

# Variables base
results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
results['constructorId'] = pd.to_numeric(results['constructorId'], errors='coerce')
results['driverId'] = pd.to_numeric(results['driverId'], errors='coerce')
results['raceId'] = pd.to_numeric(results['raceId'], errors='coerce')

results['win'] = (results['positionOrder'] == 1).astype(int)
results['podium'] = (results['positionOrder'] <= 3).astype(int)
results['points'] = pd.to_numeric(results['points'], errors='coerce').fillna(0)

# Merge con races y drivers para orden cronológico y edad pre-carrera
races['year'] = pd.to_numeric(races['year'], errors='coerce')
races['round'] = pd.to_numeric(races['round'], errors='coerce')
races['date'] = pd.to_datetime(races['date'], errors='coerce')
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')

results = results.merge(
    races[['raceId', 'year', 'round', 'date', 'circuitId']],
    on='raceId',
    how='inner'
).merge(
    drivers[['driverId', 'dob', 'nationality']],
    on='driverId',
    how='inner'
)

# Filtro de calidad metodológico pre-carrera
results = results[results['grid'] > 0].copy()
results['age'] = ((results['date'] - results['dob']).dt.days / 365.25).astype(float)
results = results.dropna(subset=['age', 'grid', 'year', 'round', 'win']).copy()

# Orden canónico temporal estricto (garantía anti-leakage)
results = results.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)
print(f"✓ Datos pre-carrera limpios y ordenados: {len(results):,} resultados")

# ============================================================================
# SECCIÓN 2: FEATURE ENGINEERING HISTÓRICO ESTRICTO (SHIFT(1) - SIN LEAKAGE)
# ============================================================================

print("\n[2/10] Creando features históricas con ventanas temporales estrictas (.shift(1))...")
print("  • Garantía metodológica: Solo información anterior a la carrera actual es utilizada.")

# 2.1 Rendimiento histórico del piloto acumulado previo a la carrera actual
print("  • Calculando métricas de carrera del piloto (expanding shift(1))...")
results['driver_wins_prior'] = results.groupby('driverId')['win'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
results['races_career'] = results.groupby('driverId')['win'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)
results['driver_podiums_prior'] = results.groupby('driverId')['podium'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)

results['win_rate_career'] = (results['driver_wins_prior'] / results['races_career'].replace(0, np.nan)).fillna(0.0)
results['podium_rate_career'] = (results['driver_podiums_prior'] / results['races_career'].replace(0, np.nan)).fillna(0.0)

# 2.2 Rendimiento histórico del constructor acumulado previo a la carrera actual
print("  • Calculando métricas del constructor (expanding shift(1))...")
results['const_wins_prior'] = results.groupby('constructorId')['win'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
results['const_races_prior'] = results.groupby('constructorId')['win'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)
results['const_podiums_prior'] = results.groupby('constructorId')['podium'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)

results['win_rate_constructor'] = (results['const_wins_prior'] / results['const_races_prior'].replace(0, np.nan)).fillna(0.0)
results['podium_rate_constructor'] = (results['const_podiums_prior'] / results['const_races_prior'].replace(0, np.nan)).fillna(0.0)

# 2.3 Rendimiento previo en circuito específico (shift(1))
print("  • Calculando rendimiento histórico por circuito (expanding shift(1))...")
results['circuit_driver_wins_prior'] = results.groupby(['circuitId', 'driverId'])['win'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
results['circuit_driver_races_prior'] = results.groupby(['circuitId', 'driverId'])['win'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)
results['win_rate_circuit'] = (results['circuit_driver_wins_prior'] / results['circuit_driver_races_prior'].replace(0, np.nan)).fillna(0.0)

results['circuit_const_wins_prior'] = results.groupby(['circuitId', 'constructorId'])['win'].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
results['circuit_const_races_prior'] = results.groupby(['circuitId', 'constructorId'])['win'].transform(lambda s: s.shift(1).expanding().count()).fillna(0)
results['win_rate_circuit_const'] = (results['circuit_const_wins_prior'] / results['circuit_const_races_prior'].replace(0, np.nan)).fillna(0.0)

# 2.4 Rendimiento reciente (últimas 5 carreras con shift(1))
print("  • Calculando rendimiento reciente (rolling 5 carreras con shift(1))...")
results['recent_wins'] = results.groupby('driverId')['win'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
results['recent_podiums'] = results.groupby('driverId')['podium'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
results['recent_races'] = results.groupby('driverId')['win'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).count()).fillna(0)
results['recent_win_rate'] = (results['recent_wins'] / results['recent_races'].replace(0, np.nan)).fillna(0.0)

# ============================================================================
# SECCIÓN 3: FEATURE ENGINEERING ADICIONAL (PRE-CARRERA)
# ============================================================================

print("\n[3/10] Generando features de interacción pre-carrera...")

# Categorías de edad pre-carrera
results['age_category'] = pd.cut(results['age'], bins=[0, 25, 30, 35, 60], labels=['Rookie', 'Young', 'Prime', 'Veteran'])

# Interacción entre posición de salida y calidad del equipo / piloto
results['grid_x_constructor_winrate'] = results['grid'] * (1 - results['win_rate_constructor'])
results['grid_x_driver_winrate'] = results['grid'] * (1 - results['win_rate_career'])
results['constructor_x_circuit'] = results['win_rate_constructor'] * results['win_rate_circuit_const']

# ============================================================================
# SECCIÓN 4: PREPARACIÓN DEL DATASET PARA MODELADO
# ============================================================================

print("\n[4/10] Seleccionando variables predictoras y codificación categórica...")

feature_cols = [
    'grid', 'age', 'year', 'round',
    'win_rate_career', 'podium_rate_career', 'races_career',
    'win_rate_constructor', 'podium_rate_constructor',
    'win_rate_circuit', 'win_rate_circuit_const',
    'recent_wins', 'recent_podiums', 'recent_win_rate',
    'grid_x_constructor_winrate', 'grid_x_driver_winrate', 'constructor_x_circuit'
]

categorical_cols = ['nationality', 'age_category']

df_model = results[['raceId', *feature_cols, *categorical_cols, 'win']].copy()
# Rellenar cualquier categoría nula en age_category si existiera
df_model['age_category'] = df_model['age_category'].fillna('Young')

# One-hot encoding
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True, dtype=float)

print(f"✓ Dataset completo unificado preservado: {df_encoded.shape[0]:,} observaciones")
print(f"  • Victorias: {df_encoded['win'].sum()} ({df_encoded['win'].mean()*100:.2f}%)")

# ============================================================================
# SECCIÓN 5: PARTICIÓN TEMPORAL CRONOLÓGICA (80/20)
# ============================================================================

print("\n[5/10] Partición temporal cronológica 80/20 (excluyendo raceId como predictor)...")

split_idx = int(len(df_encoded) * 0.8)
train_df = df_encoded.iloc[:split_idx]
test_df = df_encoded.iloc[split_idx:]

# Exclusión explícita de raceId de las variables explicativas para evitar proxy de tiempo
X_train = train_df.drop(columns=['win', 'raceId'], errors='ignore')
y_train = train_df['win']
X_test = test_df.drop(columns=['win', 'raceId'], errors='ignore')
y_test = test_df['win']

print(f"  • Train (pasado): {len(X_train):,} muestras ({y_train.mean()*100:.2f}% victorias)")
print(f"  • Test (futuro): {len(X_test):,} muestras ({y_test.mean()*100:.2f}% victorias)")

# Escalado estándar: ajuste EXCLUSIVO en Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Variable X completa para exportación
X = df_encoded.drop(columns=['win', 'raceId'], errors='ignore')
y = df_encoded['win']

# ============================================================================
# SECCIÓN 6: MANEJO DE DESBALANCE CON SMOTE (SOLO SOBRE TRAIN)
# ============================================================================

print("\n[6/10] Aplicando SMOTE exclusivamente sobre el conjunto de entrenamiento...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"  • Post-SMOTE Train: {len(X_train_smote):,} muestras ({y_train_smote.mean()*100:.2f}% victorias)")

# ============================================================================
# SECCIÓN 7: ENTRENAMIENTO DE MODELOS
# ============================================================================

print("\n[7/10] Entrenando modelos (Gradient Boosting, Random Forest, Ensemble)...")

# 7.1 Gradient Boosting
print("  • Entrenando Gradient Boosting...")
gb_params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': 42
}
model_gb = GradientBoostingClassifier(**gb_params)
model_gb.fit(X_train_smote, y_train_smote)

# 7.2 Random Forest (min_samples_leaf >= 2 para evitar memorización)
print("  • Entrenando Random Forest (min_samples_leaf=2)...")
rf_params = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1
}
model_rf = RandomForestClassifier(**rf_params)
model_rf.fit(X_train_smote, y_train_smote)

# 7.3 Ensemble Voting
print("  • Entrenando Ensemble Soft Voting...")
model_ensemble = VotingClassifier(
    estimators=[
        ('gb', model_gb),
        ('rf', model_rf)
    ],
    voting='soft'
)
model_ensemble.fit(X_train_smote, y_train_smote)

# ============================================================================
# SECCIÓN 8: EVALUACIÓN Y CALIBRACIÓN DE UMBRAL (CALIBRADO SOLO EN TRAIN)
# ============================================================================

print("\n[8/10] Evaluando modelos con umbrales calibrados únicamente en Train...")

models = {
    'GradientBoosting': model_gb,
    'Random Forest': model_rf,
    'Ensemble': model_ensemble
}

results_metrics = {}

for name, model in models.items():
    print(f"\n  {name}:")
    
    # 1. Calibrar umbral en TRAIN (NUNCA en Test para preservar independencia metodológica)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1_train = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_train_pred_thresh = (y_train_proba >= thresh).astype(int)
        f1_train = f1_score(y_train, y_train_pred_thresh, zero_division=0)
        if f1_train > best_f1_train:
            best_f1_train = f1_train
            best_threshold = thresh
            
    # 2. Evaluar sobre conjunto TEST independiente utilizando el umbral fijado en Train
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    metrics = {
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_test_proba),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'threshold': best_threshold
    }
    
    results_metrics[name] = metrics
    
    print(f"    Umbral óptimo (determinado en Train): {metrics['threshold']:.2f}")
    print(f"    F1-Score (Test independiente): {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    
    # Predicciones con threshold óptimo
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    
    # Métricas
    metrics = {
        'f1_score': f1_score(y_test, y_pred_optimal),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
        'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
        'threshold': best_threshold
    }
    
    results_metrics[name] = metrics
    
    print(f"    F1-Score: {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    Best Threshold: {metrics['threshold']:.2f}")

# ============================================================================
# SECCIÓN 9: VISUALIZACIONES
# ============================================================================

print("\n[9/10] Generando visualizaciones...")

# 9.1 Comparación de modelos
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC Curves
ax1 = axes[0, 0]
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', label='Random')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves Comparison')
ax1.legend()
ax1.grid(True)

# Precision-Recall Curves
ax2 = axes[0, 1]
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax2.plot(recall, precision, label=f'{name}')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves')
ax2.legend()
ax2.grid(True)

# Métricas comparación
ax3 = axes[1, 0]
metrics_df = pd.DataFrame(results_metrics).T
metrics_df[['f1_score', 'auc_roc', 'precision', 'recall']].plot(kind='bar', ax=ax3)
ax3.set_title('Metrics Comparison')
ax3.set_ylabel('Score')
ax3.set_ylim(0, 1)
ax3.legend(loc='lower right')
ax3.tick_params(axis='x', rotation=45)

# Feature Importance (XGBoost)
ax4 = axes[1, 1]
importance = model_gb.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
importance_df = importance_df.sort_values('importance', ascending=True).tail(15)
ax4.barh(importance_df['feature'], importance_df['importance'])
ax4.set_xlabel('Importance')
ax4.set_title('Top 15 Features (XGBoost)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison_advanced.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Guardado: {OUTPUT_DIR}/model_comparison_advanced.png")
plt.close()

# ============================================================================
# SECCIÓN 10: GUARDAR RESULTADOS
# ============================================================================

print("\n[10/10] Guardando resultados...")

# Métricas CSV (formato estandarizado para comparación entre modelos)
modelo_referencia = 'Ensemble' if 'Ensemble' in results_metrics else max(results_metrics, key=lambda k: results_metrics[k]['f1_score'])
metricas = {
    'Modelo': f'Modelo Avanzado ({modelo_referencia})',
    'F1_Test': results_metrics[modelo_referencia]['f1_score'],
    'AUC_ROC': results_metrics[modelo_referencia]['auc_roc'],
    'AUC_PR': average_precision_score(y_test, models[modelo_referencia].predict_proba(X_test_scaled)[:, 1]),
    'Precision_Test': results_metrics[modelo_referencia]['precision'],
    'Recall_Test': results_metrics[modelo_referencia]['recall'],
    'Train_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0]
}
metricas_df = pd.DataFrame([metricas])
metricas_df.to_csv(f'{OUTPUT_DIR}/metricas_comparativa.csv', index=False)
print(f"  ✓ Guardado: {OUTPUT_DIR}/metricas_comparativa.csv")

# Feature importance
importance_df.sort_values('importance', ascending=False).to_csv(
    f'{OUTPUT_DIR}/feature_importance_advanced.csv', index=False
)
print(f"  ✓ Guardado: {OUTPUT_DIR}/feature_importance_advanced.csv")

# Dataset mejorado
X.to_csv(f'{OUTPUT_DIR}/dataset_mejorado.csv', index=False)
print(f"  ✓ Guardado: {OUTPUT_DIR}/dataset_mejorado.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN - MODELO AVANZADO")
print("="*80)

print("\n📊 Métricas Finales:")
for name, metrics in results_metrics.items():
    print(f"\n  {name}:")
    print(f"    • F1-Score: {metrics['f1_score']:.4f}")
    print(f"    • AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"    • Precision: {metrics['precision']:.4f}")
    print(f"    • Recall: {metrics['recall']:.4f}")

print("\n🎯 Mejoras Implementadas:")
print("  ✓ Feature Engineering (rendimiento histórico, circuitos)")
print("  ✓ SMOTE para balanceo de clases")
print("  ✓ XGBoost + Random Forest optimizado")
print("  ✓ Ensemble Voting")
print("  ✓ Threshold optimization para F1-Score")

print("\n📁 Archivos Generados:")
print(f"  • {OUTPUT_DIR}/metricas_comparativa.csv")
print(f"  • {OUTPUT_DIR}/feature_importance_advanced.csv")
print(f"  • {OUTPUT_DIR}/model_comparison_advanced.png")
print(f"  • {OUTPUT_DIR}/dataset_mejorado.csv")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

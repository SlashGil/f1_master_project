"""
========================================
MODELO 3: PERCEPTRÓN MULTICAPA (MLP) CON UMBRAL CALIBRADO
========================================

Este script implementa un Perceptrón Multicapa (Red Neuronal) para predecir
la probabilidad de victoria en carreras de Fórmula 1.

CRITERIO METODOLÓGICO Y CALIBRACIÓN DE UMBRAL (HOMOLOGADO CON LOGIT Y RANDOM FOREST):
  - Probabilidades empíricas naturales (MLE): Optimización con Binary Cross-Entropy
    estándar no ponderada (sin distorsión de class_weight='balanced').
  - Umbral de decisión óptimo (τ*): Calibrado exclusivamente sobre el conjunto de
    entrenamiento mediante predicciones Out-Of-Fold (OOF) con 5-Fold Race-Aware TimeSeriesSplit.
  - Aislamiento temporal estricto: Cero fuga de datos; el conjunto de prueba (Test) jamás
    se utiliza para early stopping, selección de pesos ni calibración de hiperparámetros.
  - Métrica de optimización: Maximización de F1-Score para la clase victoria (win=1) en la curva PR.
  - Métrica de dominio deportivo: Precisión por carrera (ArgMax P(win) por raceId).

ARQUITECTURA DE RED:
  - Capa de entrada: Dimensionalidad numérica del dataset (features pre-carrera)
  - Capa oculta 1: 64 neuronas, activación ReLU, regularización L2 (weight decay)
  - Capa oculta 2: 32 neuronas, activación ReLU, regularización L2 (weight decay)
  - Capa de salida: 1 neurona, activación Sigmoid (probabilidad posterior P(win=1|X))

AUTOR: Salvador Romero Gil
FECHA: 2026
"""

import os
import sys
import copy
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Backend: PyTorch ({DEVICE})")
except ImportError:
    HAS_TORCH = False
    from sklearn.neural_network import MLPClassifier
    DEVICE = 'cpu (Scikit-Learn MLPClassifier)'
    print("Backend: Scikit-Learn MLPClassifier (PyTorch no instalado)")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]

DATA_DIR = "data" if os.path.exists("data") else "../data"
OUTPUT_DIR = "resultados_mlp" if os.path.exists("data") else "../resultados_mlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def race_aware_time_series_split(df_subset, n_splits=5):
    """Genera pliegues temporales agrupados por carreras completas (raceId),
    garantizando que ninguna carrera individual quede dividida entre entrenamiento y validación."""
    unique_races = df_subset["raceId"].drop_duplicates().tolist()
    n_races = len(unique_races)
    split_size = n_races // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_races = set(unique_races[:split_size * i])
        val_races = set(unique_races[split_size * i : split_size * (i + 1)])
        tr_idx = df_subset.index[df_subset["raceId"].isin(train_races)].values
        val_idx = df_subset.index[df_subset["raceId"].isin(val_races)].values
        splits.append((tr_idx, val_idx))
    return splits


if HAS_TORCH:
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            self.layer1 = nn.Linear(input_dim, 64)
            self.layer2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.sigmoid(self.output(x))
            return x


print("=" * 80)
print("MODELO 3: PERCEPTRÓN MULTICAPA (MLP) - HOMOLOGACIÓN METODOLÓGICA")
print("=" * 80)

# ============================================================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================================================

print("\n[1/8] Cargando dataset procesado...")

try:
    df = pd.read_csv(f"{DATA_DIR}/dataset_tesis_f1.csv")
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  • Variable objetivo: win")

    n_victorias = df["win"].sum()
    pct_victorias = (n_victorias / len(df)) * 100
    print(f"  • Victorias: {n_victorias} ({pct_victorias:.2f}%)")
    print(f"  • No victorias: {len(df) - n_victorias} ({100-pct_victorias:.2f}%)")
except FileNotFoundError:
    print("✗ Error: No se encontró dataset_tesis_f1.csv")
    sys.exit(1)

# ============================================================================
# SECCIÓN 2: PREPARACIÓN DE FEATURES Y TARGET
# ============================================================================

print("\n[2/8] Preparando Features y Target...")

cols_to_exclude = ["win", "raceId", "driverId", "year"] + [c for c in df.columns if c.startswith("nationality")]
X = df.drop(columns=cols_to_exclude, errors="ignore")
y = df["win"]

feature_names = X.columns.tolist()
print(f"Features seleccionados: {X.shape[1]} (sin 'year' ni 'nationality')")
print(f"Target: win (binario: 1=Victoria, 0=No Victoria)")

# ============================================================================
# SECCIÓN 3: DIVISIÓN DE DATOS (SPLIT TEMPORAL ATÓMICO)
# ============================================================================

print("\n[3/8] Dividiendo datos con corte temporal (sin aleatoriedad)...")

if {"year", "round", "raceId"}.issubset(df.columns):
    df_sorted = df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)
else:
    df_sorted = df.sort_values("raceId").reset_index(drop=True)

if {"year", "round"}.issubset(df_sorted.columns) and ((df_sorted["year"] == 2012) & (df_sorted["round"] == 18)).any():
    split_idx = df_sorted[(df_sorted["year"] == 2012) & (df_sorted["round"] == 18)].index.max() + 1
else:
    split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

X_train = train_df.drop(columns=cols_to_exclude, errors="ignore")
y_train = train_df["win"]
X_test = test_df.drop(columns=cols_to_exclude, errors="ignore")
y_test = test_df["win"]

print(f"✓ Entrenamiento (pasado): {X_train.shape[0]} muestras ({X_train.shape[0]/len(df_sorted)*100:.2f}%)")
print(f"✓ Prueba (futuro): {X_test.shape[0]} muestras ({X_test.shape[0]/len(df_sorted)*100:.2f}%)")
print("✓ Split temporal atómico aplicado (GP Abu Dabi 2012 íntegro en Train, Test inicia en GP EE.UU. 2012)")

# ============================================================================
# SECCIÓN 4: CALIBRACIÓN DE UMBRAL ÓPTIMO (5-FOLD OOF EN TRAIN)
# ============================================================================

print("\n[4/8] Calibrando umbral óptimo sin fuga (5-Fold Race-Aware TimeSeriesSplit en Train)...")
print("-" * 80)

cv_splits = race_aware_time_series_split(train_df, n_splits=5)
oof_probas, oof_targets = [], []
best_epochs_folds = []

torch.manual_seed(42)
np.random.seed(42)

for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits, 1):
    fold_scaler = StandardScaler()
    X_tr_fold = fold_scaler.fit_transform(X_train.iloc[tr_idx])
    X_va_fold = fold_scaler.transform(X_train.iloc[va_idx])
    y_tr_fold = y_train.iloc[tr_idx].values.copy()
    y_va_fold = y_train.iloc[va_idx].values.copy()

    if HAS_TORCH:
        tr_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_fold), torch.FloatTensor(y_tr_fold).unsqueeze(1)),
            batch_size=200,
            shuffle=True
        )
        X_va_tensor = torch.FloatTensor(X_va_fold).to(DEVICE)
        y_va_tensor = torch.FloatTensor(y_va_fold).unsqueeze(1).to(DEVICE)

        torch.manual_seed(42 + fold_idx)
        fold_model = MLP(X_train.shape[1]).to(DEVICE)
        optimizer = optim.Adam(fold_model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()

        best_va_loss = float("inf")
        best_epoch = 0
        patience = 10
        patience_counter = 0
        best_state = None

        for ep in range(50):
            fold_model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = fold_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            fold_model.eval()
            with torch.no_grad():
                va_out = fold_model(X_va_tensor)
                va_loss = criterion(va_out, y_va_tensor).item()

            if va_loss < best_va_loss:
                best_va_loss = va_loss
                best_epoch = ep + 1
                best_state = copy.deepcopy(fold_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        best_epochs_folds.append(best_epoch)
        fold_model.load_state_dict(best_state)
        fold_model.eval()
        with torch.no_grad():
            va_proba = fold_model(X_va_tensor).cpu().numpy().flatten()
    else:
        fold_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=100,
            random_state=42 + fold_idx
        )
        fold_model.fit(X_tr_fold, y_tr_fold)
        va_proba = fold_model.predict_proba(X_va_fold)[:, 1]
        best_epochs_folds.append(fold_model.n_iter_)

    oof_probas.append(va_proba)
    oof_targets.append(y_va_fold)
    print(f"  • Pliegue {fold_idx}/5 completado (Época óptima: {best_epochs_folds[-1]})")

oof_probas = np.concatenate(oof_probas)
oof_targets = np.concatenate(oof_targets)

oof_prec, oof_rec, oof_thresh = precision_recall_curve(oof_targets, oof_probas)
oof_f1 = 2 * oof_prec[:-1] * oof_rec[:-1] / (oof_prec[:-1] + oof_rec[:-1] + 1e-12)
best_idx = int(np.argmax(oof_f1))
best_threshold = float(oof_thresh[best_idx])
best_oof_f1 = float(oof_f1[best_idx])

print(f"\n✓ Umbral óptimo calibrado (F1-max OOF Train): τ* = {best_threshold:.6f} (F1 OOF = {best_oof_f1:.4f})")
optimal_final_epochs = int(np.median(best_epochs_folds))
print(f"✓ Épocas óptimas seleccionadas para modelo final (mediana OOF): {optimal_final_epochs}")

# ============================================================================
# SECCIÓN 5: ENTRENAMIENTO DEL MODELO FINAL
# ============================================================================

print("\n[5/8] Entrenando modelo final de MLP sobre todo Train...")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

history = {"loss": []}
start_time = time.time()

if HAS_TORCH:
    full_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.values.copy()).unsqueeze(1)
        ),
        batch_size=200,
        shuffle=True
    )
    torch.manual_seed(42)
    final_model = MLP(X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for ep in range(optimal_final_epochs):
        final_model.train()
        ep_loss = 0.0
        n_batches = 0
        for bx, by in full_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = final_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        history["loss"].append(ep_loss / max(1, n_batches))

    training_time = time.time() - start_time
    final_model.eval()
    with torch.no_grad():
        y_train_pred_proba = final_model(torch.FloatTensor(X_train_scaled).to(DEVICE)).cpu().numpy().flatten()
        y_test_pred_proba = final_model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy().flatten()
else:
    final_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=optimal_final_epochs,
        random_state=42
    )
    final_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    history["loss"] = final_model.loss_curve_
    y_train_pred_proba = final_model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]

print(f"✓ Modelo entrenado en {training_time:.2f} segundos ({len(history['loss'])} épocas)")

# ============================================================================
# SECCIÓN 6: PREDICCIONES CON UMBRAL CALIBRADO
# ============================================================================

print("\n[6/8] Realizando predicciones con umbral calibrado...")

y_train_pred_binary = (y_train_pred_proba >= best_threshold).astype(int)
y_test_pred_binary = (y_test_pred_proba >= best_threshold).astype(int)

# Evaluación por carrera (ArgMax: piloto con mayor P(win) en cada GP)
test_eval_df = test_df[["raceId", "win"]].copy()
test_eval_df["pred_proba"] = y_test_pred_proba
carreras_acertadas = 0
total_carreras = 0
for rid, group in test_eval_df.groupby("raceId"):
    idx_max = group["pred_proba"].idxmax()
    if group.loc[idx_max, "win"] == 1:
        carreras_acertadas += 1
    total_carreras += 1

acc_carrera = (carreras_acertadas / total_carreras * 100) if total_carreras > 0 else 0.0

print(f"✓ Umbral aplicado: τ* = {best_threshold:.6f}")
print(f"✓ Positivos predichos en Test: {int(y_test_pred_binary.sum())}")
print(f"✓ Precisión por Carrera (ArgMax): {acc_carrera:.2f}% ({carreras_acertadas}/{total_carreras} GPs acertados)")

# ============================================================================
# SECCIÓN 7: EVALUACIÓN DE MÉTRICAS Y CSV COMPARATIVO
# ============================================================================

print("\n[7/8] Evaluando métricas del modelo...")
print("-" * 80)

f1_train = f1_score(y_train, y_train_pred_binary)
f1_test = f1_score(y_test, y_test_pred_binary)
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
pr_auc = average_precision_score(y_test, y_test_pred_proba)
test_precision = precision_score(y_test, y_test_pred_binary)
test_recall = recall_score(y_test, y_test_pred_binary)
test_acc = accuracy_score(y_test, y_test_pred_binary)

print(f"  • F1-Score Train:                    {f1_train:.4f}")
print(f"  • F1-Score Test:                     {f1_test:.4f}")
print(f"  • F1-Score OOF (Train):              {best_oof_f1:.4f}")
print(f"  • AUC-ROC Test:                      {roc_auc:.4f}")
print(f"  • AUC-PR Test:                       {pr_auc:.4f}")
print(f"  • Precisión Test:                    {test_precision:.4f}")
print(f"  • Recall Test:                       {test_recall:.4f}")
print(f"  • Accuracy Test:                     {test_acc:.4f}")
print(f"  • ArgMax Top-1 Accuracy:             {acc_carrera:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_test_pred_binary,
        target_names=["No Victoria", "Victoria"],
        digits=4
    )
)

# Guardar métricas comparativas homologadas
metricas = {
    "Modelo": "Perceptron Multicapa (MLP)",
    "F1_Test": f1_test,
    "AUC_ROC": roc_auc,
    "AUC_PR": pr_auc,
    "Precision_Test": test_precision,
    "Recall_Test": test_recall,
    "Accuracy_Test": test_acc,
    "Threshold_OOF": best_threshold,
    "F1_OOF_Train": best_oof_f1,
    "Precision_Carrera_ArgMax": acc_carrera,
    "ArgMax_Top1_Acc": acc_carrera / 100.0,
    "GPs_Acertados": carreras_acertadas,
    "Total_GPs": total_carreras,
    "Train_Size": X_train.shape[0],
    "Test_Size": X_test.shape[0],
}
pd.DataFrame([metricas]).to_csv(f"{OUTPUT_DIR}/metricas_comparativa.csv", index=False)
print(f"✓ Métricas guardadas en {OUTPUT_DIR}/metricas_comparativa.csv")

# ============================================================================
# SECCIÓN 8: ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ============================================================================

print("\n[8/8] Analizando importancia de características y generando gráficos...")
print("-" * 80)

if HAS_TORCH:
    weights = final_model.layer1.weight.detach().cpu().numpy()
else:
    weights = final_model.coefs_[0].T

importance = np.mean(np.abs(weights), axis=0)

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)
feature_importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
print(f"✓ Importancia guardada en {OUTPUT_DIR}/feature_importance.csv")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

# 1) Top 20 Feature Importance
plt.figure(figsize=(14, 10))
top_features = feature_importance_df.head(20)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
plt.barh(range(len(top_features)), top_features["importance"], color=colors, edgecolor="black")
plt.yticks(range(len(top_features)), [f[:45] + "..." if len(f) > 45 else f for f in top_features["feature"]])
plt.xlabel("Importancia (Media de |Pesos| Primera Capa)", fontsize=12, fontweight="bold")
plt.title("Top 20 Variables Más Determinantes - Perceptrón Multicapa (MLP)", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 2) Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="#D1495B", lw=3, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Clasificador Aleatorio (AUC = 0.5)")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=12, fontweight="bold")
plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)", fontsize=12, fontweight="bold")
plt.title("Curva ROC - Perceptrón Multicapa (MLP)", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 3) Curva Precision-Recall con umbral calibrado marcado
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color="#2E86AB", lw=3, label=f"PR (AUC = {pr_auc:.4f})")
plt.scatter([test_recall], [test_precision], color="#D1495B", s=120, zorder=5,
            label=f"Punto operativo τ*={best_threshold:.4f} (P={test_precision:.3f}, R={test_recall:.3f}, F1={f1_test:.3f})")
plt.axhline(float(y_test.mean()), color="gray", linestyle="--", linewidth=1.5,
            label=f"Prevalencia Test = {y_test.mean():.4f}")
plt.xlabel("Recall", fontsize=12, fontweight="bold")
plt.ylabel("Precisión", fontsize=12, fontweight="bold")
plt.title("Curva Precision-Recall - Perceptrón Multicapa (MLP)", fontsize=14, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/precision_recall_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 4) Matriz de Confusión
cm = confusion_matrix(y_test, y_test_pred_binary)
fig, ax = plt.subplots(figsize=(10, 8))
if sns is not None:
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples_r",
        annot_kws={"size": 14},
        cbar_kws={"label": "Número de Predicciones"},
        xticklabels=["Pred: No Victoria", "Pred: Victoria"],
        yticklabels=["Real: No Victoria", "Real: Victoria"],
        ax=ax,
    )
else:
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    fig.colorbar(im, ax=ax, label="Número de Predicciones")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Victoria", "Pred: Victoria"], fontsize=12)
    ax.set_yticklabels(["Real: No Victoria", "Real: Victoria"], fontsize=12)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

plt.xlabel("Predicción", fontsize=12, fontweight="bold")
plt.ylabel("Valor Real", fontsize=12, fontweight="bold")
plt.title(f"Matriz de Confusión (τ* = {best_threshold:.4f}) - MLP", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 5) Curva de Entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(history["loss"], label="Pérdida de Entrenamiento (BCE)", color="#2E86AB", lw=2)
plt.xlabel("Época", fontsize=12, fontweight="bold")
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12, fontweight="bold")
plt.title("Evolución del Entrenamiento - Perceptrón Multicapa (MLP)", fontsize=14, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_history_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 6) Predicciones vs Reales
plt.figure(figsize=(12, 8))
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
plt.scatter(
    y_test_pred_proba[sample_indices],
    y_test.iloc[sample_indices],
    alpha=0.4,
    s=25,
    color="#2E86AB",
    edgecolors="black",
    linewidth=0.3
)
plt.plot([0, 1], [0, 1], "r--", lw=2, label="Línea de Identidad")
plt.axvline(x=best_threshold, color="green", linestyle="--", lw=2, label=f"Umbral óptimo τ* = {best_threshold:.4f}")
plt.axvline(x=0.5, color="orange", linestyle=":", lw=1.5, label="Umbral estándar 0.5")
plt.xlabel("Probabilidad Predicha", fontsize=12, fontweight="bold")
plt.ylabel("Valor Real", fontsize=12, fontweight="bold")
plt.title("Predicciones vs Reales - Perceptrón Multicapa (MLP)", fontsize=14, fontweight="bold")
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/predictions_vs_actual_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# 7) Distribución de Importancia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.hist(feature_importance_df["importance"], bins=30, color="#2E86AB", alpha=0.7, edgecolor="black")
ax1.axvline(
    feature_importance_df["importance"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mediana: {feature_importance_df['importance'].median():.6f}"
)
ax1.set_xlabel("Importancia", fontsize=11, fontweight="bold")
ax1.set_ylabel("Frecuencia", fontsize=11, fontweight="bold")
ax1.set_title("Distribución de Importancia de Variables", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

bp = ax2.boxplot(feature_importance_df["importance"], patch_artist=True)
bp["boxes"][0].set_facecolor("#D1495B")
bp["boxes"][0].set_alpha(0.7)
ax2.set_ylabel("Importancia", fontsize=11, fontweight="bold")
ax2.set_title("Boxplot de Importancia de Variables", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/importance_distribution_mlp.png", dpi=300, bbox_inches="tight")
plt.close()

# Guardar modelo y transformadores
if HAS_TORCH:
    torch.save(final_model.state_dict(), f"{OUTPUT_DIR}/modelo_mlp.pt")
    print(f"\n💾 Pesos de red guardados en: {OUTPUT_DIR}/modelo_mlp.pt")
else:
    joblib.dump(final_model, f"{OUTPUT_DIR}/modelo_mlp.joblib")
    print(f"\n💾 Modelo Scikit-Learn guardado en: {OUTPUT_DIR}/modelo_mlp.joblib")

joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.joblib")
print(f"💾 Scaler guardado en: {OUTPUT_DIR}/scaler.joblib")

print("\n" + "=" * 80)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 80)

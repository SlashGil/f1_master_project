"""
========================================
MODELO 2: RANDOM FOREST (BASELINE + GRID SEARCH)
========================================

Este script entrena y evalúa dos variantes de Random Forest:
1) Baseline con hiperparámetros fijos
2) Tuning con GridSearchCV

Cada variante genera sus propios artefactos en carpetas separadas.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = [12, 8]

DATA_DIR = "data" if os.path.exists("data") else "../data"
OUTPUT_BASELINE = "resultados_random_forest_baseline" if os.path.exists("data") else "../resultados_random_forest_baseline"
OUTPUT_GRID = "resultados_random_forest_grid_search" if os.path.exists("data") else "../resultados_random_forest_grid_search"

os.makedirs(OUTPUT_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_GRID, exist_ok=True)

print("=" * 80)
print("MODELO 2: RANDOM FOREST - BASELINE + GRID SEARCH")
print("=" * 80)

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

print("\n[1/4] Cargando dataset procesado...")

df = pd.read_csv(f"{DATA_DIR}/dataset_tesis_f1.csv")
print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

print("\n[2/4] Aplicando split temporal (80/20)...")

# 1. Ordenar el dataframe cronológicamente
if {"year", "round", "raceId"}.issubset(df.columns):
    df_sorted = df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)
else:
    df_sorted = df.sort_values("raceId").reset_index(drop=True)

# 2. Definir features (X) y target (y)
# Se excluye 'raceId' (identificador), 'win' (target) y 'year'/'nationality_*' (fuera de tesis).
cols_to_exclude = ["win", "raceId", "year"] + [c for c in df_sorted.columns if c.startswith("nationality")]
X = df_sorted.drop(columns=cols_to_exclude, errors="ignore")
y = df_sorted["win"]
print(f"Features seleccionados: {X.shape[1]} variables (sin 'year' ni 'nationality')")

# 3. Realizar el split temporal atómico (GP Abu Dabi 2012 íntegro en Train)
if {"year", "round"}.issubset(df_sorted.columns) and ((df_sorted["year"] == 2012) & (df_sorted["round"] == 18)).any():
    split_idx = df_sorted[(df_sorted["year"] == 2012) & (df_sorted["round"] == 18)].index.max() + 1
else:
    split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(df_sorted)*100:.2f}%) | Test: {X_test.shape[0]} muestras ({X_test.shape[0]/len(df_sorted)*100:.2f}%)")
print("✓ Split temporal atómico: GP Abu Dabi 2012 100% en Train, Test inicia en GP EE.UU. 2012")


# ============================================================================
# FUNCIONES
# ============================================================================

def evaluar_y_guardar(model, output_dir, model_label, feature_names, best_params=None, best_cv_f1=None):
    """Evalúa modelo y guarda métricas + visualizaciones en output_dir."""

    print("\n" + "-" * 80)
    print(f"Evaluando: {model_label}")
    print("-" * 80)

    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred_binary = model.predict(X_train)
    y_test_pred_binary = model.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred_binary)
    f1_test = f1_score(y_test, y_test_pred_binary)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    pr_auc = average_precision_score(y_test, y_test_pred_proba)
    test_precision = precision_score(y_test, y_test_pred_binary)
    test_recall = recall_score(y_test, y_test_pred_binary)

    print(f"F1 Train: {f1_train:.4f}")
    print(f"F1 Test: {f1_test:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred_binary,
            target_names=["No Victoria", "Victoria"],
            digits=4,
        )
    )

    # CSV métricas comparables
    metricas = {
        "Modelo": model_label,
        "F1_Test": f1_test,
        "AUC_ROC": roc_auc,
        "AUC_PR": pr_auc,
        "Precision_Test": test_precision,
        "Recall_Test": test_recall,
        "Train_Size": X_train.shape[0],
        "Test_Size": X_test.shape[0],
    }
    pd.DataFrame([metricas]).to_csv(f"{output_dir}/metricas_comparativa.csv", index=False)

    # CSV parámetros del modelo
    if best_params is None:
        best_params = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "class_weight": model.class_weight,
        }

    params_row = {
        "Modelo": model_label,
        "Best_CV_F1": best_cv_f1 if best_cv_f1 is not None else np.nan,
        **best_params,
    }
    pd.DataFrame([params_row]).to_csv(f"{output_dir}/best_params_random_forest.csv", index=False)

    # Feature importance
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    # 1) Top feature importance
    plt.figure(figsize=(14, 10))
    top_features = feature_importance_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_features["importance"], color=colors, edgecolor="black")
    plt.yticks(range(len(top_features)), [f[:45] + "..." if len(f) > 45 else f for f in top_features["feature"]])
    plt.xlabel("Importance")
    plt.title(f"Top 20 Variables - {model_label}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="#D1495B", lw=3, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_label}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(recall_vals, precision_vals, color="#2E86AB", lw=3, label=f"PR (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_label}")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4) Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred_binary)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges_r",
        annot_kws={"size": 14},
        cbar_kws={"label": "Número de Predicciones"},
        xticklabels=["Pred: No Victoria", "Pred: Victoria"],
        yticklabels=["Real: No Victoria", "Real: Victoria"],
    )
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title(f"Matriz de Confusión - {model_label}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5) Predicciones vs reales
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
        linewidth=0.3,
    )
    plt.plot([0, 1], [0, 1], "r--", lw=2, label="Línea identidad")
    plt.axhline(y=0.5, color="orange", linestyle=":", lw=2, label="Umbral 0.5")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Valor real")
    plt.title(f"Predicciones vs Reales - {model_label}")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_vs_actual_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6) Distribución de importancia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(feature_importance_df["importance"], bins=30, color="#2E86AB", alpha=0.7, edgecolor="black")
    ax1.axvline(
        feature_importance_df["importance"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mediana: {feature_importance_df['importance'].median():.6f}",
    )
    ax1.set_title("Distribución de Importancia")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    bp = ax2.boxplot(feature_importance_df["importance"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#D1495B")
    bp["boxes"][0].set_alpha(0.7)
    ax2.set_title("Boxplot de Importancia")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/importance_distribution_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7) Importancia por tipo
    numeric_importance = feature_importance_df[
        feature_importance_df["feature"].apply(lambda x: "_" not in x)
    ]["importance"].sum()
    constructor_importance = feature_importance_df[
        feature_importance_df["feature"].str.contains("constructorId_")
    ]["importance"].sum()

    tipos = ["Variables Numéricas (grid, age, round)", "ConstructorID"]
    importancias = [numeric_importance, constructor_importance]
    colores = ["#2E86AB", "#D1495B"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(tipos, importancias, color=colores, alpha=0.7, edgecolor="black")
    plt.xlabel("Tipo de variable")
    plt.ylabel("Importancia acumulada")
    plt.title(f"Importancia por Tipo - {model_label}")
    for bar, imp in zip(bars, importancias):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{imp:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/importance_by_type_rf.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Resultados guardados en: {output_dir}")


# ============================================================================
# ENTRENAMIENTO BASELINE
# ============================================================================

print("\n[3/4] Entrenando Random Forest baseline...")
baseline_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
baseline_model.fit(X_train, y_train)

baseline_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
}


evaluar_y_guardar(
    model=baseline_model,
    output_dir=OUTPUT_BASELINE,
    model_label="Random Forest (Baseline)",
    feature_names=X.columns,
    best_params=baseline_params,
    best_cv_f1=None,
)


# ============================================================================
# ENTRENAMIENTO GRID SEARCH
# ============================================================================

print("\n[4/4] Entrenando Random Forest con GridSearchCV...")

base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [8, 10, 12],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [2, 4, 8],
}

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

cv_strategy = race_aware_time_series_split(train_df, n_splits=5)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)
grid_model = grid_search.best_estimator_

print(f"Mejor F1 CV: {grid_search.best_score_:.4f}")
print(f"Mejores parámetros: {grid_search.best_params_}")

evaluar_y_guardar(
    model=grid_model,
    output_dir=OUTPUT_GRID,
    model_label="Random Forest (GridSearch)",
    feature_names=X.columns,
    best_params=grid_search.best_params_,
    best_cv_f1=grid_search.best_score_,
)

print("\n" + "=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)
print(f"Carpeta baseline: {OUTPUT_BASELINE}")
print(f"Carpeta tuning:   {OUTPUT_GRID}")

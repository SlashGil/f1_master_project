"""
========================================
MODELO 2: RANDOM FOREST (BASELINE + GRID SEARCH CON UMBRAL CALIBRADO)
========================================

Este script entrena y evalúa dos variantes de Random Forest:
1) Baseline con hiperparámetros fijos
2) Tuning con GridSearchCV y validación temporal por carreras (Race-Aware TimeSeriesSplit)

CRITERIO METODOLÓGICO Y CALIBRACIÓN DE UMBRAL (HOMOLOGADO CON REGRESIÓN LOGÍSTICA):
  - class_weight=None: Probabilidades empíricas naturales sin distorsión de pesos artificiales.
  - Umbral de decisión óptimo (τ*): Calibrado exclusivamente sobre el conjunto de entrenamiento
    mediante predicciones Out-Of-Fold (OOF) con 5-Fold Race-Aware TimeSeriesSplit (sin fuga de datos).
  - Métrica de optimización: Maximización de F1-Score para la clase victoria (win=1) en la curva PR.
  - Evaluación complementaria de dominio: Precisión por carrera (ArgMax P(win) por raceId).

Cada variante genera sus propios artefactos en carpetas separadas.
"""

import os
import sys
import tempfile
import warnings

# Configuración de codificación para terminales Windows
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

# Evitar advertencias de directorio de cache de matplotlib y problemas con GUI en Windows/multiprocessing
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = [12, 8]

DATA_DIR = "data" if os.path.exists("data") else "../data"
OUTPUT_BASELINE = "resultados_random_forest_baseline" if os.path.exists("data") else "../resultados_random_forest_baseline"
OUTPUT_GRID = "resultados_random_forest_grid_search" if os.path.exists("data") else "../resultados_random_forest_grid_search"


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


def calibrar_umbral_f1_oof(model_params, X_train, y_train, train_df, n_splits=5):
    """
    Calibra el umbral óptimo de decisión sin fuga de datos mediante Out-Of-Fold (OOF).
    Aplica n_splits de Race-Aware TimeSeriesSplit sobre Train, generando predicciones
    de probabilidad para cada fold de validación temporal y seleccionando el umbral
    tau* que maximiza el F1-Score sobre la clase minoritaria (win=1).
    """
    print("\n🎯 Calibrando umbral óptimo sin fuga mediante 5-Fold Race-Aware TimeSeriesSplit en Train...")
    cv_splits = race_aware_time_series_split(train_df, n_splits=n_splits)
    oof_probas, oof_targets = [], []

    fold_params = dict(model_params)
    if fold_params.get("class_weight") == "None":
        fold_params["class_weight"] = None

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits, 1):
        fold_model = RandomForestClassifier(
            **fold_params,
            random_state=42,
            n_jobs=-1,
        )
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        va_proba = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_probas.append(va_proba)
        oof_targets.append(y_train.iloc[va_idx].values)

    oof_probas = np.concatenate(oof_probas)
    oof_targets = np.concatenate(oof_targets)

    oof_prec, oof_rec, oof_thresh = precision_recall_curve(oof_targets, oof_probas)
    oof_f1 = 2 * oof_prec[:-1] * oof_rec[:-1] / (oof_prec[:-1] + oof_rec[:-1] + 1e-12)
    best_idx = int(np.argmax(oof_f1))
    best_threshold = float(oof_thresh[best_idx])
    best_oof_f1 = float(oof_f1[best_idx])

    print(f"✓ Umbral óptimo calibrado (F1-max OOF Train): τ* = {best_threshold:.6f} (F1 OOF = {best_oof_f1:.4f})")
    return best_threshold, best_oof_f1


def evaluar_y_guardar(
    model,
    output_dir,
    model_label,
    feature_names,
    X_train,
    y_train,
    X_test,
    y_test,
    test_df,
    best_params=None,
    best_cv_score=None,
    threshold=0.5,
    oof_f1=None,
):
    """Evalúa modelo y guarda métricas + visualizaciones en output_dir aplicando umbral calibrado."""
    print("\n" + "-" * 80)
    print(f"Evaluando: {model_label}")
    print("-" * 80)

    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Binarización con umbral calibrado tau* (homologado con Regresión Logística)
    y_train_pred_binary = (y_train_pred_proba >= threshold).astype(int)
    y_test_pred_binary = (y_test_pred_proba >= threshold).astype(int)

    f1_train = f1_score(y_train, y_train_pred_binary)
    f1_test = f1_score(y_test, y_test_pred_binary)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    pr_auc = average_precision_score(y_test, y_test_pred_proba)
    test_precision = precision_score(y_test, y_test_pred_binary)
    test_recall = recall_score(y_test, y_test_pred_binary)
    test_acc = accuracy_score(y_test, y_test_pred_binary)

    # Evaluación de precisión por carrera (ArgMax: piloto con mayor P(win) en cada GP)
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

    print(f"Umbral aplicado (τ*): {threshold:.6f}")
    if oof_f1 is not None:
        print(f"F1 OOF (Train): {oof_f1:.4f}")
    print(f"Accuracy Test: {test_acc:.4f}")
    print(f"F1 Train: {f1_train:.4f}")
    print(f"F1 Test: {f1_test:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR: {pr_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Precisión por Carrera (ArgMax): {acc_carrera:.2f}% ({carreras_acertadas}/{total_carreras} GPs ganados acertados)")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred_binary,
            target_names=["No Victoria", "Victoria"],
            digits=4,
        )
    )

    # CSV métricas comparables homologadas
    metricas = {
        "Modelo": model_label,
        "F1_Test": f1_test,
        "AUC_ROC": roc_auc,
        "AUC_PR": pr_auc,
        "Precision_Test": test_precision,
        "Recall_Test": test_recall,
        "Accuracy_Test": test_acc,
        "Threshold_OOF": threshold,
        "F1_OOF_Train": oof_f1 if oof_f1 is not None else np.nan,
        "Precision_Carrera_ArgMax": acc_carrera,
        "ArgMax_Top1_Acc": acc_carrera / 100.0,
        "GPs_Acertados": carreras_acertadas,
        "Total_GPs": total_carreras,
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
        "Best_CV_Score": best_cv_score if best_cv_score is not None else np.nan,
        "Threshold_OOF": threshold,
        "F1_OOF_Train": oof_f1 if oof_f1 is not None else np.nan,
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
    plt.scatter([test_recall], [test_precision], color="#D1495B", s=120, zorder=5,
                label=f"Punto operativo τ*={threshold:.4f} (P={test_precision:.3f}, R={test_recall:.3f}, F1={f1_test:.3f})")
    plt.axhline(float(y_test.mean()), color="gray", linestyle="--", linewidth=1.5,
                label=f"Prevalencia Test = {y_test.mean():.4f}")
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
    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges_r",
            annot_kws={"size": 14},
            cbar_kws={"label": "Número de Predicciones"},
            xticklabels=["Pred: No Victoria", "Pred: Victoria"],
            yticklabels=["Real: No Victoria", "Real: Victoria"],
            ax=ax,
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Oranges")
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
    plt.axvline(x=threshold, color="green", linestyle="--", lw=2, label=f"Umbral óptimo τ* = {threshold:.4f}")
    plt.axvline(x=0.5, color="orange", linestyle=":", lw=1.5, label="Umbral estándar 0.5")
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


def main():
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

    print("\n[2/4] Aplicando split temporal atómico (Abu Dabi 2012)...")

    # 1. Ordenar el dataframe cronológicamente
    if {"year", "round", "raceId"}.issubset(df.columns):
        df_sorted = df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)
    else:
        df_sorted = df.sort_values("raceId").reset_index(drop=True)

    # 2. Definir features (X) y target (y)
    # Se excluye 'raceId' (identificador), 'win' (target) y 'year'/'nationality_*' (fuera de tesis).
    cols_to_exclude = ["win", "raceId", "driverId", "year"] + [c for c in df_sorted.columns if c.startswith("nationality")]
    X = df_sorted.drop(columns=cols_to_exclude, errors="ignore")
    y = df_sorted["win"]
    print(f"Features seleccionados: {X.shape[1]} variables (sin 'year' ni 'nationality')")

    # 3. Realizar el split temporal atómico (GP Abu Dabi 2012 íntegro en Train)
    if {"year", "round"}.issubset(df_sorted.columns) and ((df_sorted["year"] == 2012) & (df_sorted["round"] == 18)).any():
        split_idx = df_sorted[(df_sorted["year"] == 2012) & (df_sorted["round"] == 18)].index.max() + 1
    else:
        split_idx = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = X.iloc[:split_idx].copy().reset_index(drop=True)
    y_train = y.iloc[:split_idx].copy().reset_index(drop=True)
    X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
    y_test = y.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(df_sorted)*100:.2f}%) | Test: {X_test.shape[0]} muestras ({X_test.shape[0]/len(df_sorted)*100:.2f}%)")
    print("✓ Split temporal atómico: GP Abu Dabi 2012 100% en Train, Test inicia en GP EE.UU. 2012")

    # ============================================================================
    # ENTRENAMIENTO BASELINE
    # ============================================================================

    print("\n[3/4] Entrenando Random Forest baseline...")
    baseline_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": None,
    }

    print("\nCalibrando umbral óptimo para Baseline (OOF en Train)...")
    baseline_threshold, baseline_oof_f1 = calibrar_umbral_f1_oof(
        baseline_params, X_train, y_train, train_df, n_splits=5
    )

    baseline_model = RandomForestClassifier(
        **baseline_params,
        random_state=42,
        n_jobs=-1,
    )
    baseline_model.fit(X_train, y_train)

    evaluar_y_guardar(
        model=baseline_model,
        output_dir=OUTPUT_BASELINE,
        model_label="Random Forest (Baseline)",
        feature_names=X.columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        best_params=baseline_params,
        best_cv_score=None,
        threshold=baseline_threshold,
        oof_f1=baseline_oof_f1,
    )

    # ============================================================================
    # ENTRENAMIENTO GRID SEARCH
    # ============================================================================

    print("\n[4/4] Entrenando Random Forest con GridSearchCV...")

    base_model = RandomForestClassifier(
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 10, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4, 8],
    }

    cv_strategy = race_aware_time_series_split(train_df, n_splits=5)

    # Nota metodológica: en datos desbalanceados con class_weight=None, scoring="roc_auc"
    # permite seleccionar los hiperparámetros con mayor capacidad discriminativa de probabilidades,
    # sin quedar sesgados por el umbral rígido estándar de 0.5.
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    grid_model = grid_search.best_estimator_

    print(f"\nMejor AUC-ROC CV: {grid_search.best_score_:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")

    print("\nCalibrando umbral óptimo para Random Forest (GridSearch)...")
    grid_threshold, grid_oof_f1 = calibrar_umbral_f1_oof(
        grid_search.best_params_, X_train, y_train, train_df, n_splits=5
    )

    evaluar_y_guardar(
        model=grid_model,
        output_dir=OUTPUT_GRID,
        model_label="Random Forest (GridSearch)",
        feature_names=X.columns,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        best_params=grid_search.best_params_,
        best_cv_score=grid_search.best_score_,
        threshold=grid_threshold,
        oof_f1=grid_oof_f1,
    )

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)
    print(f"Carpeta baseline: {OUTPUT_BASELINE}")
    print(f"Carpeta tuning:   {OUTPUT_GRID}")


if __name__ == "__main__":
    main()

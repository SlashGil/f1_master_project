"""
==============================================================================
VARIANTE ABLACIÓN: REGRESIÓN LOGÍSTICA UNIVARIADA (SOLO 'grid')
==============================================================================

Objetivo:
  Evaluar la capacidad predictiva de la Regresión Logística utilizando
  únicamente la posición efectiva de salida en parrilla ('grid', 1 feature),
  bajo exactamente el mismo protocolo causal y sin fuga que el modelo de 201
  variables (4a y 4_regresion_logistica.py).

Esto permite cuantificar:
  1) El poder predictivo basal de la posición de salida por sí sola.
  2) La ganancia marginal aportada por constructorId_*, age y round (201 vs 1).
"""

from __future__ import annotations

import os
import sys
import time
import warnings

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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_tesis_f1.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados_regresion_logistica_solo_grid")
FULL_MODEL_METRICS_PATH = os.path.join(PROJECT_ROOT, "resultados_regresion_logistica", "metricas_modelo.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = [10, 7]
plt.rcParams["font.size"] = 10


def race_aware_time_series_split(df_subset: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera pliegues temporales expansivos agrupados por carreras completas (raceId)."""
    unique_races = df_subset["raceId"].drop_duplicates().tolist()
    split_size = len(unique_races) // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_races = set(unique_races[: split_size * i])
        val_races = set(unique_races[split_size * i : split_size * (i + 1)])
        tr_idx = df_subset.index[df_subset["raceId"].isin(train_races)].values
        val_idx = df_subset.index[df_subset["raceId"].isin(val_races)].values
        splits.append((tr_idx, val_idx))
    return splits


def calibrate_oof_threshold(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | None,
) -> tuple[float, float, dict[str, float]]:
    """Calibra el umbral F1-max mediante 5-Fold Race-Aware TimeSeriesSplit en Train."""
    oof_probas, oof_targets = [], []
    fold_metrics = []
    for tr_idx, va_idx in race_aware_time_series_split(train_df, n_splits=5):
        fold_scaler = StandardScaler()
        X_tr_f = fold_scaler.fit_transform(X_train.iloc[tr_idx])
        X_va_f = fold_scaler.transform(X_train.iloc[va_idx])
        y_tr_f = y_train.iloc[tr_idx]
        y_va_f = y_train.iloc[va_idx]

        clf = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        clf.fit(X_tr_f, y_tr_f)
        p_va = clf.predict_proba(X_va_f)[:, 1]
        oof_probas.append(p_va)
        oof_targets.append(y_va_f.values)
        fold_metrics.append(
            {
                "auc_roc": float(roc_auc_score(y_va_f, p_va)),
                "auc_pr": float(average_precision_score(y_va_f, p_va)),
                "log_loss": float(log_loss(y_va_f, p_va)),
                "brier": float(brier_score_loss(y_va_f, p_va)),
            }
        )

    oof_p = np.concatenate(oof_probas)
    oof_y = np.concatenate(oof_targets)
    prec, rec, thresh = precision_recall_curve(oof_y, oof_p)
    f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_vals))
    cv_means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    return float(thresh[best_idx]), float(f1_vals[best_idx]), cv_means


def evaluate_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    class_weight: str | None,
    label: str,
) -> tuple[dict, LogisticRegression, np.ndarray, np.ndarray]:
    """Entrena y evalúa la variante univariada (sin umbral y con umbral OOF)."""
    tau_oof, f1_oof, cv_means = calibrate_oof_threshold(train_df, X_train, y_train, class_weight)

    t0 = time.time()
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    elapsed = time.time() - t0

    p_train = model.predict_proba(X_train_scaled)[:, 1]
    p_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (p_test >= tau_oof).astype(int)

    # Evaluación complementaria ArgMax por carrera (en solo-grid equivale a menor grid)
    eval_df = pd.DataFrame(
        {
            "raceId": test_df["raceId"].values,
            "grid": test_df["grid"].values,
            "proba": p_test,
            "win": y_test.values,
        }
    )
    top1_idx = eval_df.groupby("raceId")["proba"].idxmax().values
    y_argmax = np.zeros(len(eval_df), dtype=int)
    y_argmax[top1_idx] = 1
    argmax_hits = int((y_argmax & y_test.values).sum())
    argmax_races = int(eval_df["raceId"].nunique())

    res = {
        "Variante": label,
        "class_weight": str(class_weight),
        "N_Features": 1,
        "Features": "grid",
        "Train_Size": len(X_train),
        "Test_Size": len(X_test),
        "AUC_ROC_Test": float(roc_auc_score(y_test, p_test)),
        "AUC_PR_Test": float(average_precision_score(y_test, p_test)),
        "Log_Loss_Test": float(log_loss(y_test, p_test)),
        "Brier_Test": float(brier_score_loss(y_test, p_test)),
        "CV_AUC_ROC_Mean": cv_means["auc_roc"],
        "CV_AUC_PR_Mean": cv_means["auc_pr"],
        "CV_Log_Loss_Mean": cv_means["log_loss"],
        "Threshold_OOF": tau_oof,
        "F1_OOF_Train": f1_oof,
        "F1_Test": float(f1_score(y_test, y_pred_test)),
        "Precision_Test": float(precision_score(y_test, y_pred_test)),
        "Recall_Test": float(recall_score(y_test, y_pred_test)),
        "Accuracy_Test": float(accuracy_score(y_test, y_pred_test)),
        "Positivos_Predichos": int(y_pred_test.sum()),
        "ArgMax_Hits": argmax_hits,
        "ArgMax_Races": argmax_races,
        "ArgMax_Top1_Acc": argmax_hits / argmax_races,
        "Beta_grid": float(model.coef_[0][0]),
        "Odds_Ratio_grid": float(np.exp(model.coef_[0][0])),
        "Intercepto": float(model.intercept_[0]),
        "Iteraciones": int(model.n_iter_[0]),
        "Tiempo_s": float(elapsed),
    }
    return res, model, p_test, y_pred_test


def main() -> None:
    print("=" * 88)
    print("VARIANTE ABLACIÓN: REGRESIÓN LOGÍSTICA SOLO CON 'grid' (1 PREDICTOR)")
    print("=" * 88)

    df = pd.read_csv(DATA_PATH)
    df_sorted = df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)

    if ((df_sorted["year"] == 2012) & (df_sorted["round"] == 18)).any():
        split_idx = int(df_sorted[(df_sorted["year"] == 2012) & (df_sorted["round"] == 18)].index.max() + 1)
    else:
        split_idx = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

    X_train = train_df[["grid"]].copy()
    y_train = train_df["win"]
    X_test = test_df[["grid"]].copy()
    y_test = test_df["win"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_solo_grid.joblib"))

    print(f"\n[1/5] Dataset: {len(df_sorted):,} filas | Predictor único: ['grid']")
    print(f"      Train: {len(X_train):,} ({len(X_train)/len(df_sorted)*100:.2f}%) | Test: {len(X_test):,} ({len(X_test)/len(df_sorted)*100:.2f}%)")

    # Evaluar MLE (principal) y Balanced (referencia)
    res_mle, model_mle, p_test_mle, y_pred_mle = evaluate_variant(
        train_df, test_df, X_train, y_train, X_test, y_test,
        X_train_scaled, X_test_scaled, None, "Solo grid (class_weight=None, MLE)"
    )
    res_bal, _model_bal, p_test_bal, _y_pred_bal = evaluate_variant(
        train_df, test_df, X_train, y_train, X_test, y_test,
        X_train_scaled, X_test_scaled, "balanced", "Solo grid (class_weight='balanced')"
    )
    joblib.dump(model_mle, os.path.join(OUTPUT_DIR, "modelo_lr_solo_grid_mle.joblib"))

    print("\n[2/5] Resultados Solo 'grid' (MLE calibrado + umbral OOF en Train):")
    print(f"  • Parámetros: β₀ = {res_mle['Intercepto']:.6f} | β_grid = {res_mle['Beta_grid']:.6f} | OR_grid = {res_mle['Odds_Ratio_grid']:.6f}")
    print(f"  • Sin umbral -> AUC-ROC: {res_mle['AUC_ROC_Test']:.4f} | AUC-PR: {res_mle['AUC_PR_Test']:.4f} | Log-Loss: {res_mle['Log_Loss_Test']:.4f} | Brier: {res_mle['Brier_Test']:.4f}")
    print(f"  • Umbral OOF -> τ* = {res_mle['Threshold_OOF']:.6f} (F1 OOF={res_mle['F1_OOF_Train']:.4f})")
    print(f"  • Test (τ*)  -> F1: {res_mle['F1_Test']:.4f} | Prec: {res_mle['Precision_Test']:.4f} | Rec: {res_mle['Recall_Test']:.4f} | Acc: {res_mle['Accuracy_Test']:.4f} | Pos: {res_mle['Positivos_Predichos']}")
    print(f"  • ArgMax/GP  -> {res_mle['ArgMax_Hits']}/{res_mle['ArgMax_Races']} ({res_mle['ArgMax_Top1_Acc']*100:.2f}%)")

    print("\n[Classification Report - Solo grid (MLE + τ* OOF)]")
    print(classification_report(y_test, y_pred_mle, target_names=["No Victoria", "Victoria"], digits=4))

    # Guardar métricas y tabla de probabilidades por posición de parrilla
    metrics_df = pd.DataFrame([res_mle, res_bal])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metricas_solo_grid.csv"), index=False)

    coef_df = pd.DataFrame(
        [
            {
                "variable": "intercepto",
                "coeficiente": res_mle["Intercepto"],
                "odds_ratio": np.exp(res_mle["Intercepto"]),
            },
            {
                "variable": "grid",
                "coeficiente": res_mle["Beta_grid"],
                "odds_ratio": res_mle["Odds_Ratio_grid"],
            },
        ]
    )
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "tabla_coeficientes_solo_grid.csv"), index=False, float_format="%.6f")

    preds_df = pd.DataFrame(
        {
            "raceId": test_df["raceId"].values,
            "driverId": test_df["driverId"].values,
            "year": test_df["year"].values,
            "round": test_df["round"].values,
            "grid": test_df["grid"].values,
            "win": y_test.values,
            "proba_solo_grid_mle": p_test_mle,
            "proba_solo_grid_balanced": p_test_bal,
            "pred_umbral_oof": y_pred_mle,
        }
    )
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "predicciones_test_solo_grid.csv"), index=False)

    # Curva empírica vs predicha por posición de parrilla (1..24)
    grid_vals = np.arange(1, 25)
    grid_scaled = scaler.transform(pd.DataFrame({"grid": grid_vals}))
    grid_probs = model_mle.predict_proba(grid_scaled)[:, 1]
    emp_test = test_df.groupby("grid").agg(n_test=("win", "count"), wins_test=("win", "sum"), rate_test=("win", "mean")).reset_index()
    emp_train = train_df.groupby("grid").agg(n_train=("win", "count"), rate_train=("win", "mean")).reset_index()
    grid_table = pd.DataFrame({"grid": grid_vals, "proba_predicha_mle": grid_probs})
    grid_table = grid_table.merge(emp_train, on="grid", how="left").merge(emp_test, on="grid", how="left")
    grid_table.to_csv(os.path.join(OUTPUT_DIR, "probabilidad_por_grid.csv"), index=False)

    # Comparativa directa contra el modelo completo de 201 variables si existe
    if os.path.exists(FULL_MODEL_METRICS_PATH):
        full_m = pd.read_csv(FULL_MODEL_METRICS_PATH).iloc[0]
        comp_df = pd.DataFrame(
            [
                {
                    "Modelo": "Solo grid (1 feature)",
                    "N_Features": 1,
                    "AUC_ROC": res_mle["AUC_ROC_Test"],
                    "AUC_PR": res_mle["AUC_PR_Test"],
                    "LogLoss": res_mle["Log_Loss_Test"],
                    "Brier": res_mle["Brier_Test"],
                    "Threshold_OOF": res_mle["Threshold_OOF"],
                    "F1_Test": res_mle["F1_Test"],
                    "Precision_Test": res_mle["Precision_Test"],
                    "Recall_Test": res_mle["Recall_Test"],
                    "ArgMax_Top1_Acc": res_mle["ArgMax_Top1_Acc"],
                },
                {
                    "Modelo": "Completo (201 features)",
                    "N_Features": int(full_m["N_Coeficientes"]),
                    "AUC_ROC": float(full_m["AUC_ROC"]),
                    "AUC_PR": float(full_m["AUC_PR"]),
                    "LogLoss": float(full_m["LogLoss_Test"]),
                    "Brier": float(full_m["Brier_Test"]),
                    "Threshold_OOF": float(full_m["Threshold_OOF"]),
                    "F1_Test": float(full_m["F1_Test"]),
                    "Precision_Test": float(full_m["Precision_Test"]),
                    "Recall_Test": float(full_m["Recall_Test"]),
                    "ArgMax_Top1_Acc": float(full_m["ArgMax_Top1_Acc"]),
                },
            ]
        )
        comp_df.to_csv(os.path.join(OUTPUT_DIR, "comparativa_201_vs_solo_grid.csv"), index=False)
        print("\n[3/5] Comparativa directa 1 feature (Solo grid) vs 201 features:")
        print(comp_df.to_string(index=False))

    print("\n[4/5] Generando gráficas de la variante Solo 'grid'...")
    # 1. ROC
    fpr, tpr, _ = roc_curve(y_test, p_test_mle)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="#2E86AB", lw=2.5, label=f"Solo grid (AUC = {res_mle['AUC_ROC_Test']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Azar (AUC = 0.5000)")
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    plt.title("Curva ROC — Regresión Logística Univariada (Solo grid)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_curva_roc_solo_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. PR Curve + umbral
    prec_c, rec_c, _ = precision_recall_curve(y_test, p_test_mle)
    plt.figure(figsize=(10, 8))
    plt.plot(rec_c, prec_c, color="#2E86AB", lw=2.5, label=f"Solo grid PR (AUC-PR = {res_mle['AUC_PR_Test']:.4f})")
    plt.scatter(
        [res_mle["Recall_Test"]],
        [res_mle["Precision_Test"]],
        color="#D1495B",
        s=120,
        zorder=5,
        label=f"Umbral OOF τ*={res_mle['Threshold_OOF']:.4f} (P={res_mle['Precision_Test']:.3f}, R={res_mle['Recall_Test']:.3f}, F1={res_mle['F1_Test']:.3f})",
    )
    plt.axhline(float(y_test.mean()), color="gray", linestyle="--", lw=1.5, label=f"Prevalencia Test = {y_test.mean():.4f}")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall — Regresión Logística Solo 'grid'")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_curva_pr_solo_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Probabilidad predicha vs tasa empírica por posición de parrilla
    plt.figure(figsize=(11, 7))
    plt.plot(grid_table["grid"], grid_table["proba_predicha_mle"], "o-", color="#2E86AB", lw=2.5, label="Probabilidad predicha (Sigmoide MLE)")
    plt.bar(grid_table["grid"], grid_table["rate_test"].fillna(0), alpha=0.35, color="#D1495B", label="Tasa empírica de victoria en Test (2012–2024)")
    plt.axhline(res_mle["Threshold_OOF"], color="#28A745", linestyle="--", lw=2, label=f"Umbral OOF τ* = {res_mle['Threshold_OOF']:.4f}")
    plt.xticks(grid_vals)
    plt.xlabel("Posición en Parrilla de Salida (grid)")
    plt.ylabel("Probabilidad / Tasa de Victoria")
    plt.title("Curva Sigmoide de Victoria por Posición de Parrilla — Solo 'grid'")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_probabilidad_vs_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_mle)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Número de Predicciones")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Victoria", "Pred: Victoria"])
    ax.set_yticklabels(["Real: No Victoria", "Real: Victoria"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > thresh else "black", fontweight="bold",
            )
    ax.grid(False)
    ax.set_xlabel("Predicción del Modelo")
    ax.set_ylabel("Valor Real")
    ax.set_title(f"Matriz de Confusión Solo 'grid' (τ* = {res_mle['Threshold_OOF']:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_matriz_confusion_solo_grid.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[5/5] Resultados guardados en: {OUTPUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()

"""
==============================================================================
PASO 1: REGRESIÓN LOGÍSTICA — EVALUACIÓN SIN UMBRAL (THRESHOLD-INDEPENDENT)
==============================================================================

Script reproducible para el Paso 1 del análisis de Regresión Logística:
  - Evalúa únicamente métricas independientes del umbral de decisión:
      * AUC-ROC (capacidad discriminativa de ranking)
      * AUC-PR / Average Precision (rendimiento en clase positiva desbalanceada)
      * Log-Loss (calidad y calibración probabilística)
      * Brier Score (error cuadrático medio probabilístico)
  - Compara de forma homogénea dos formulaciones:
      1) MLE estándar (class_weight=None): probabilidades calibradas al prior real
      2) Ponderada (class_weight='balanced'): pesos inversos a frecuencia de clase
  - Ejecuta validación cruzada temporal por carreras completas (5-Fold Race-Aware
    TimeSeriesSplit) dentro de Train antes de evaluar en el conjunto Test ciego.
  - Persiste modelos (.joblib), escalador, predicciones de probabilidad en Test,
    tablas de coeficientes/Odds Ratios, métricas y gráficas en:
      resultados_regresion_logistica_sin_umbral/

Garantías anti-leakage:
  - Ordenación cronológica estricta ['year', 'round', 'raceId']
  - Partición atómica en GP Abu Dabi 2012 (20,020 Train / 5,013 Test)
  - Exclusión explícita de 'win', 'raceId', 'driverId', 'year' y 'nationality_*' (201 features)
  - StandardScaler ajustado únicamente en entrenamiento (tanto global como por fold)
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
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_tesis_f1.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados_regresion_logistica_sin_umbral")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = [10, 7]
plt.rcParams["font.size"] = 10

MODEL_VARIANTS = [
    ("MLE_Sin_Pesos", "Regresión Logística (class_weight=None)", None, "#2E86AB"),
    ("Balanced", "Regresión Logística (class_weight='balanced')", "balanced", "#D1495B"),
]


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


def compute_threshold_free_metrics(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calcula métricas estrictamente independientes del umbral de decisión."""
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def evaluate_temporal_cv(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str | None,
    variant_key: str,
) -> pd.DataFrame:
    """Ejecuta 5-Fold Race-Aware TimeSeriesSplit dentro de Train ajustando scaler por fold."""
    cv_rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(race_aware_time_series_split(train_df, n_splits=5), start=1):
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
        proba_va = clf.predict_proba(X_va_f)[:, 1]
        metrics_va = compute_threshold_free_metrics(y_va_f, proba_va)
        cv_rows.append(
            {
                "Variante": variant_key,
                "Fold": fold_idx,
                "N_Train_Fold": len(tr_idx),
                "N_Val_Fold": len(va_idx),
                "Wins_Val_Fold": int(y_va_f.sum()),
                "AUC_ROC": metrics_va["auc_roc"],
                "AUC_PR": metrics_va["auc_pr"],
                "Log_Loss": metrics_va["log_loss"],
                "Brier_Score": metrics_va["brier_score"],
            }
        )
    return pd.DataFrame(cv_rows)


def main() -> None:
    print("=" * 88)
    print("PASO 1: REGRESIÓN LOGÍSTICA SIN UMBRAL (MÉTRICAS INDEPENDIENTES DEL UMBRAL)")
    print("=" * 88)

    df = pd.read_csv(DATA_PATH)
    df_sorted = df.sort_values(["year", "round", "raceId"]).reset_index(drop=True)

    cols_to_exclude = ["win", "raceId", "driverId", "year"] + [
        c for c in df_sorted.columns if c.startswith("nationality")
    ]

    if ((df_sorted["year"] == 2012) & (df_sorted["round"] == 18)).any():
        split_idx = int(df_sorted[(df_sorted["year"] == 2012) & (df_sorted["round"] == 18)].index.max() + 1)
    else:
        split_idx = int(len(df_sorted) * 0.8)

    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)

    X_train = train_df.drop(columns=cols_to_exclude, errors="ignore")
    y_train = train_df["win"]
    X_test = test_df.drop(columns=cols_to_exclude, errors="ignore")
    y_test = test_df["win"]
    feature_names = X_train.columns.tolist()

    print(f"\n[1/5] Dataset cargado: {len(df_sorted):,} filas | Features en X: {len(feature_names)}")
    print(f"      Train: {len(train_df):,} filas ({len(train_df)/len(df_sorted)*100:.2f}%) | Victorias: {int(y_train.sum())} ({y_train.mean()*100:.2f}%)")
    print(f"      Test : {len(test_df):,} filas ({len(test_df)/len(df_sorted)*100:.2f}%) | Victorias: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

    metrics_records = []
    cv_all = []
    test_probas: dict[str, np.ndarray] = {}

    print("\n[2/5] Validación cruzada temporal (5-Fold Race-Aware TimeSeriesSplit) y ajuste final...")
    for variant_key, variant_label, cw, _color in MODEL_VARIANTS:
        cv_df = evaluate_temporal_cv(train_df, X_train, y_train, cw, variant_key)
        cv_all.append(cv_df)

        t0 = time.time()
        model = LogisticRegression(
            class_weight=cw,
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)
        elapsed = time.time() - t0

        p_train = model.predict_proba(X_train_scaled)[:, 1]
        p_test = model.predict_proba(X_test_scaled)[:, 1]
        test_probas[variant_key] = p_test

        m_train = compute_threshold_free_metrics(y_train, p_train)
        m_test = compute_threshold_free_metrics(y_test, p_test)

        metrics_records.append(
            {
                "Variante": variant_key,
                "Descripcion": variant_label,
                "class_weight": str(cw),
                "N_Features": len(feature_names),
                "Train_Size": len(X_train),
                "Test_Size": len(X_test),
                "AUC_ROC_Test": m_test["auc_roc"],
                "AUC_PR_Test": m_test["auc_pr"],
                "Log_Loss_Test": m_test["log_loss"],
                "Brier_Score_Test": m_test["brier_score"],
                "AUC_ROC_Train": m_train["auc_roc"],
                "AUC_PR_Train": m_train["auc_pr"],
                "Log_Loss_Train": m_train["log_loss"],
                "Brier_Score_Train": m_train["brier_score"],
                "CV_AUC_ROC_Mean": float(cv_df["AUC_ROC"].mean()),
                "CV_AUC_PR_Mean": float(cv_df["AUC_PR"].mean()),
                "CV_Log_Loss_Mean": float(cv_df["Log_Loss"].mean()),
                "CV_Brier_Mean": float(cv_df["Brier_Score"].mean()),
                "Intercepto": float(model.intercept_[0]),
                "Iteraciones": int(model.n_iter_[0]),
                "Tiempo_s": float(elapsed),
            }
        )

        # Guardar modelo y tabla de coeficientes
        model_fname = "modelo_lr_mle.joblib" if cw is None else "modelo_lr_balanced.joblib"
        joblib.dump(model, os.path.join(OUTPUT_DIR, model_fname))

        coef_df = pd.DataFrame(
            {
                "variable": feature_names,
                "coeficiente": model.coef_[0],
                "odds_ratio": np.exp(model.coef_[0]),
                "abs_coeficiente": np.abs(model.coef_[0]),
            }
        ).sort_values("abs_coeficiente", ascending=False)
        coef_fname = "tabla_coeficientes_mle.csv" if cw is None else "tabla_coeficientes_balanced.csv"
        coef_df.to_csv(os.path.join(OUTPUT_DIR, coef_fname), index=False, float_format="%.6f")

        print(f"\n  • {variant_label}:")
        print(f"      CV (5-Fold Train) -> AUC-ROC: {cv_df['AUC_ROC'].mean():.4f} | AUC-PR: {cv_df['AUC_PR'].mean():.4f} | Log-Loss: {cv_df['Log_Loss'].mean():.4f} | Brier: {cv_df['Brier_Score'].mean():.4f}")
        print(f"      Test Ciego        -> AUC-ROC: {m_test['auc_roc']:.4f} | AUC-PR: {m_test['auc_pr']:.4f} | Log-Loss: {m_test['log_loss']:.4f} | Brier: {m_test['brier_score']:.4f}")
        print(f"      Rango proba Test  -> [{p_test.min():.6f}, {p_test.max():.6f}] | Convergió en {int(model.n_iter_[0])} iteraciones")

    print("\n[3/5] Guardando tablas de métricas y predicciones probabilísticas...")
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metricas_sin_umbral.csv"), index=False)

    cv_full_df = pd.concat(cv_all, ignore_index=True)
    cv_full_df.to_csv(os.path.join(OUTPUT_DIR, "cv_temporal_sin_umbral.csv"), index=False)

    preds_df = pd.DataFrame(
        {
            "raceId": test_df["raceId"].values,
            "driverId": test_df["driverId"].values,
            "year": test_df["year"].values,
            "round": test_df["round"].values,
            "grid": test_df["grid"].values,
            "win": y_test.values,
            "proba_mle_none": test_probas["MLE_Sin_Pesos"],
            "proba_balanced": test_probas["Balanced"],
        }
    )
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "predicciones_test_sin_umbral.csv"), index=False)

    print("\n[4/5] Generando gráficas independientes del umbral...")
    # 1. Curva ROC
    plt.figure(figsize=(10, 8))
    for variant_key, variant_label, _cw, color in MODEL_VARIANTS:
        p_te = test_probas[variant_key]
        fpr, tpr, _ = roc_curve(y_test, p_te)
        auc_val = roc_auc_score(y_test, p_te)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{variant_label} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Azar (AUC = 0.5000)")
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    plt.title("Curva ROC (Sin Umbral) — Regresión Logística")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_curva_roc_sin_umbral.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Curva Precision-Recall
    plt.figure(figsize=(10, 8))
    for variant_key, variant_label, _cw, color in MODEL_VARIANTS:
        p_te = test_probas[variant_key]
        prec, rec, _ = precision_recall_curve(y_test, p_te)
        ap_val = average_precision_score(y_test, p_te)
        plt.plot(rec, prec, color=color, lw=2.5, label=f"{variant_label} (AUC-PR = {ap_val:.4f})")
    baseline_prev = float(y_test.mean())
    plt.axhline(baseline_prev, color="gray", linestyle="--", lw=1.5, label=f"Prevalencia Test = {baseline_prev:.4f}")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall (Sin Umbral) — Regresión Logística")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_curva_pr_sin_umbral.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Curva de Calibración (Reliability Diagram)
    plt.figure(figsize=(10, 8))
    for variant_key, variant_label, _cw, color in MODEL_VARIANTS:
        p_te = test_probas[variant_key]
        prob_true, prob_pred = calibration_curve(y_test, p_te, n_bins=10, strategy="quantile")
        brier_val = brier_score_loss(y_test, p_te)
        plt.plot(prob_pred, prob_true, marker="o", color=color, lw=2, label=f"{variant_label} (Brier = {brier_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Calibración Perfecta")
    plt.xlabel("Probabilidad Media Predicha")
    plt.ylabel("Fracción Real de Victorias")
    plt.title("Calibración Probabilística en Test — Regresión Logística")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_calibracion_probabilidades.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Distribución de probabilidades por clase real
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for ax, (variant_key, variant_label, _cw, _color) in zip(axes, MODEL_VARIANTS):
        p_te = test_probas[variant_key]
        ax.hist(p_te[y_test == 0], bins=30, alpha=0.6, density=True, color="#6C757D", label="No Victoria (win=0)")
        ax.hist(p_te[y_test == 1], bins=30, alpha=0.6, density=True, color="#2E86AB", label="Victoria (win=1)")
        ax.set_title(variant_label)
        ax.set_xlabel("Probabilidad Predicha P(win=1)")
        ax.set_ylabel("Densidad")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Distribución de Probabilidades Predichas en Test (Sin Umbral)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_distribucion_probabilidades.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[5/5] Artefactos guardados en: {OUTPUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()


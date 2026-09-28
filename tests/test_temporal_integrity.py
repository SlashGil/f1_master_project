"""
==============================================================================
TEST DE INTEGRIDAD TEMPORAL Y PREVENCIÓN DE FUGA DE INFORMACIÓN (DATA LEAKAGE)
==============================================================================

Propósito:
  Verificar rigurosamente que:
  1. En el split temporal 80/20, el conjunto de prueba (Test) contiene
     únicamente carreras con fecha mayor o igual a la última carrera del
     conjunto de entrenamiento (Train):
         max(fecha_train) <= min(fecha_test)
  2. En particiones TimeSeriesSplit (validación cruzada temporal), el bloque de
     validación de cada fold utiliza estrictamente carreras posteriores en el
     tiempo a las carreras de su bloque de entrenamiento:
         max(fecha_train_k) <= min(fecha_val_k)
  3. No exista fuga por inclusión de 'raceId' como variable predictora en los
     modelos estadísticos y de Machine Learning.
  4. No existan variables post-carrera o intra-carrera en el dataset unificado.

Compatibilidad:
  Utiliza la biblioteca estándar de Python (csv, datetime, unittest) para
  garantizar ejecución sin dependencias externas en cualquier entorno o CI/CD.
"""

import os
import re
import csv
import unittest
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELOS_DIR = os.path.join(PROJECT_ROOT, "modelos")

DATASET_CSV = os.path.join(DATA_DIR, "dataset_tesis_f1.csv")
RACES_CSV = os.path.join(DATA_DIR, "races.csv")


class TestTemporalIntegrity(unittest.TestCase):
    """Pruebas unitarias de integridad cronológica y anti-leakage."""

    @classmethod
    def setUpClass(cls):
        """Cargar catálogo de fechas de carreras y dataset procesado."""
        if not os.path.exists(RACES_CSV):
            raise FileNotFoundError(f"No se encontró races.csv en {RACES_CSV}")
        if not os.path.exists(DATASET_CSV):
            raise FileNotFoundError(f"No se encontró dataset_tesis_f1.csv en {DATASET_CSV}")

        # 1. Mapa de metadatos de carreras por raceId
        cls.races_meta = {}
        with open(RACES_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                race_id = int(row["raceId"])
                date_str = row.get("date")
                race_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
                cls.races_meta[race_id] = {
                    "date": race_date,
                    "year": int(row["year"]) if row.get("year") else None,
                    "round": int(row["round"]) if row.get("round") else None,
                    "name": row.get("name", "")
                }

        # 2. Cargar filas del dataset_tesis_f1.csv manteniendo su orden físico
        cls.dataset_rows = []
        cls.headers = []
        with open(DATASET_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cls.headers = reader.fieldnames or []
            for row in reader:
                race_id = int(row["raceId"])
                meta = cls.races_meta.get(race_id)
                cls.dataset_rows.append({
                    "raceId": race_id,
                    "year": int(float(row["year"])) if "year" in row else meta["year"],
                    "round": int(float(row["round"])) if "round" in row else meta["round"],
                    "date": meta["date"] if meta else None,
                    "win": int(float(row["win"])) if "win" in row else None
                })

    def test_01_chronological_split_boundary(self):
        """
        Validar que max(fecha_train) <= min(fecha_test) en la partición 80/20.
        Garantiza que ninguna carrera del conjunto de prueba ocurrió antes que
        las carreras del conjunto de entrenamiento.
        """
        total_rows = len(self.dataset_rows)
        self.assertGreater(total_rows, 0, "El dataset no debe estar vacío")

        split_idx = int(total_rows * 0.8)
        train_slice = self.dataset_rows[:split_idx]
        test_slice = self.dataset_rows[split_idx:]

        # Extraer fechas válidas
        train_dates = [r["date"] for r in train_slice if r["date"] is not None]
        test_dates = [r["date"] for r in test_slice if r["date"] is not None]

        self.assertTrue(train_dates, "El conjunto train debe contener fechas válidas")
        self.assertTrue(test_dates, "El conjunto test debe contener fechas válidas")

        max_train_date = max(train_dates)
        min_test_date = min(test_dates)

        # Aserción central de no contaminación temporal
        self.assertLessEqual(
            max_train_date,
            min_test_date,
            f"Fuga temporal detectada: La fecha máxima de train ({max_train_date.strftime('%Y-%m-%d')}) "
            f"es posterior a la fecha mínima de test ({min_test_date.strftime('%Y-%m-%d')})"
        )

        # Reportar frontera cronológica validada
        print(f"\n  [OK] Frontera Temporal 80/20 Validada:")
        print(f"       • Train (80% Pasado): {len(train_slice):,} filas | Fecha máx: {max_train_date.strftime('%Y-%m-%d')}")
        print(f"       • Test  (20% Futuro): {len(test_slice):,} filas | Fecha mín: {min_test_date.strftime('%Y-%m-%d')}")

    def test_02_time_series_split_expanding_folds(self):
        """
        Validar que en cualquier partición TimeSeriesSplit (k pliegues),
        cada bloque de validación utiliza carreras estrictamente posteriores
        al bloque de entrenamiento de ese pliegue:
            max(fecha_train_k) <= min(fecha_val_k)
        """
        total_rows = len(self.dataset_rows)
        n_splits = 5
        fold_size = total_rows // (n_splits + 1)

        print(f"\n  [OK] Validando {n_splits} folds de TimeSeriesSplit:")

        for i in range(1, n_splits + 1):
            train_end = fold_size * i
            val_end = train_end + fold_size

            train_fold = self.dataset_rows[:train_end]
            val_fold = self.dataset_rows[train_end:val_end]

            train_dates = [r["date"] for r in train_fold if r["date"] is not None]
            val_dates = [r["date"] for r in val_fold if r["date"] is not None]

            max_train_d = max(train_dates)
            min_val_d = min(val_dates)

            self.assertLessEqual(
                max_train_d,
                min_val_d,
                f"Fallo en Fold {i}: max(fecha_train)={max_train_d.strftime('%Y-%m-%d')} "
                f"> min(fecha_val)={min_val_d.strftime('%Y-%m-%d')}"
            )

            print(f"       • Fold {i}: Train ({len(train_fold):,} filas, máx {max_train_d.strftime('%Y-%m-%d')}) "
                  f"<= Val ({len(val_fold):,} filas, mín {min_val_d.strftime('%Y-%m-%d')})")

    def test_03_no_prohibited_in_race_features(self):
        """
        Verificar que el dataset no contenga variables intra-carrera o post-carrera
        (fastestLapSpeed, milliseconds_pit_stop, milliseconds_lap_time, positionOrder, points).
        """
        prohibited_features = {
            "fastestlapspeed",
            "fastestlaptime",
            "milliseconds_pit_stop",
            "milliseconds_lap_time",
            "positionorder",
            "points",
            "laps",
            "rank",
            "statusid"
        }

        dataset_cols_lower = {col.lower() for col in self.headers}
        found_prohibited = dataset_cols_lower.intersection(prohibited_features)

        self.assertEqual(
            found_prohibited,
            set(),
            f"Se encontraron variables con fuga temporal en dataset_tesis_f1.csv: {found_prohibited}"
        )
        print(f"\n  [OK] Verificado: Cero variables post-carrera prohibidas en el dataset.")

    def test_04_raceid_excluded_from_model_features(self):
        """
        Inspeccionar el código fuente de los scripts de modelado para garantizar
        que 'raceId' se elimina explícitamente de la matriz de predictores X, X_train y X_test.
        """
        scripts_to_check = [
            "1_regresion_lineal.py",
            "2_random_forest.py",
            "3_perceptron_multicapa.py",
            "4_regresion_logistica.py"
        ]

        print(f"\n  [OK] Verificando exclusión de 'raceId' en scripts de modelado:")

        for script_name in scripts_to_check:
            script_path = os.path.join(MODELOS_DIR, script_name)
            self.assertTrue(os.path.exists(script_path), f"No existe {script_path}")

            with open(script_path, mode="r", encoding="utf-8") as f:
                code = f.read()

            # Buscar que se descarte 'raceId' en X o en drop
            has_raceid_drop = re.search(
                r"drop\s*\(\s*(columns\s*=\s*)?\[[^\]]*['\"]raceId['\"][^\]]*\]",
                code,
                re.IGNORECASE
            )

            self.assertIsNotNone(
                has_raceid_drop,
                f"El script {script_name} NO excluye explícitamente 'raceId' de la matriz de características X."
            )
            print(f"       • {script_name:<26} -> Excluye 'raceId' explícitamente.")

    def test_05_unified_dataset_size(self):
        """
        Verificar que el dataset unificado oficial conserve exactamente
        las 25,121 observaciones requeridas por la auditoría académica.
        """
        expected_rows = 25121
        actual_rows = len(self.dataset_rows)
        self.assertEqual(
            actual_rows,
            expected_rows,
            f"Tamaño inesperado del dataset: {actual_rows} (esperado: {expected_rows})"
        )
        print(f"\n  [OK] Integridad Muestral: 25,121 observaciones confirmadas.")


if __name__ == "__main__":
    unittest.main()

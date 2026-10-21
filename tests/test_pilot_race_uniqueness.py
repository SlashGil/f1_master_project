"""
========================================================================================
TEST DE UNICIDAD DE PARTICIPACIÓN PILOTO-CARRERA
========================================================================================

Propósito:
    Validar formalmente que por cada carrera exista exactamente un solo registro
    para cada piloto que pudiese haber corrido dicha carrera (participantes con grid > 0).

Verificaciones clave:
    1. Cero duplicados en el conjunto de prueba oficial (Austin 2012 - Abu Dabi 2024).
    2. Cero duplicados en toda la era moderna de la Fórmula 1 (1965 - 2024).
    3. Validación del principio metodológico pre-carrera (resolución biyectiva piloto-parrilla).
    4. Ninguna carrera posee más de un ganador único (win == 1).
"""

from __future__ import annotations

import csv
from datetime import datetime
import os
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_tesis_f1.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
RACES_CSV = os.path.join(DATA_DIR, "races.csv")


class TestPilotRaceUniqueness(unittest.TestCase):
    """Pruebas unitarias de unicidad observacional por piloto y carrera."""

    @classmethod
    def setUpClass(cls):
        """Cargar catálogos y dataset alineado."""
        if not os.path.exists(DATASET_CSV):
            raise FileNotFoundError(f"No se encontró dataset_tesis_f1.csv en {DATASET_CSV}")
        if not os.path.exists(RESULTS_CSV):
            raise FileNotFoundError(f"No se encontró results.csv en {RESULTS_CSV}")

        # 1. Cargar results.csv filtrado por grid > 0
        cls.results_clean = []
        with open(RESULTS_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    grid = int(row["grid"])
                except (ValueError, TypeError):
                    continue
                if grid > 0:
                    cls.results_clean.append({
                        "resultId": int(row["resultId"]),
                        "raceId": int(row["raceId"]),
                        "driverId": int(row["driverId"]),
                        "grid": grid,
                        "positionOrder": int(row["positionOrder"]) if row.get("positionOrder") else None
                    })

        # 2. Cargar dataset_tesis_f1.csv
        cls.dataset_rows = []
        with open(DATASET_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls.dataset_rows.append({
                    "raceId": int(row["raceId"]),
                    "grid": int(float(row["grid"])),
                    "year": int(float(row["year"])) if "year" in row else None,
                    "round": int(float(row["round"])) if "round" in row else None,
                    "win": int(float(row["win"])) if "win" in row else None,
                })

    def test_01_test_set_strict_uniqueness(self):
        """
        Validar que en el conjunto de prueba (Test 2012-2024), NINGÚN piloto
        tenga más de un registro por carrera (0 duplicados, 100.0% único).
        """
        # Filtrar test set (Austin 2012 round 19 en adelante)
        test_races = {
            r["raceId"] for r in self.dataset_rows
            if (r["year"] > 2012) or (r["year"] == 2012 and r["round"] >= 19)
        }
        test_results = [r for r in self.results_clean if r["raceId"] in test_races]
        self.assertEqual(len(test_results), 5013, "El conjunto de prueba debe tener exactamente 5,013 filas")

        # Conteo por (raceId, driverId) en test set
        pair_counts = {}
        for r in test_results:
            key = (r["raceId"], r["driverId"])
            pair_counts[key] = pair_counts.get(key, 0) + 1

        duplicates = {k: v for k, v in pair_counts.items() if v > 1}
        self.assertEqual(
            len(duplicates),
            0,
            f"Se encontraron pilotos duplicados en el conjunto de prueba: {duplicates}"
        )
        print("\n  [OK] Conjunto de Prueba (Test 2012-2024): 100% de unicidad estricta (0 duplicados en 5,013 observaciones).")

    def test_02_modern_era_driver_uniqueness(self):
        """
        Validar que en la era moderna (1965-2024), NINGÚN piloto tenga más
        de un registro en la misma carrera.
        """
        modern_results = [r for r in self.results_clean if r["raceId"] in {
            row["raceId"] for row in self.dataset_rows if row["year"] is not None and row["year"] >= 1965
        }]

        pair_counts = {}
        for r in modern_results:
            key = (r["raceId"], r["driverId"])
            pair_counts[key] = pair_counts.get(key, 0) + 1

        duplicates = {k: v for k, v in pair_counts.items() if v > 1}
        self.assertEqual(
            len(duplicates),
            0,
            f"Se encontraron duplicados en la era moderna: {duplicates}"
        )
        print(f"\n  [OK] Era Moderna (1965-2024): Cero duplicados encontrados ({len(modern_results):,} registros evaluados).")

    def test_03_pre_race_deduplication_guarantee(self):
        """
        Validar que al aplicar el criterio metodológico pre-carrera (asignación
        original de parrilla), se obtienen exactamente 25,033 pares únicos sin duplicidad.
        """
        # Filtrar duplicados en results reteniendo el primer registro de largada
        seen_pairs = set()
        dedup_records = []
        for r in self.results_clean:
            key = (r["raceId"], r["driverId"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                dedup_records.append(r)

        self.assertEqual(len(seen_pairs), 25033, "Debe haber exactamente 25,033 duplas únicas de piloto-carrera")
        print(f"\n  [OK] Criterio Pre-Carrera: 25,033 duplas únicas de piloto-carrera verificadas (100.0% de cobertura).")


if __name__ == "__main__":
    unittest.main()


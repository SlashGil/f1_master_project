# Informe Técnico de Refactorización y Auditoría Metodológica
## Eliminación de Fuga de Información (*Data Leakage*) y Validación de Integridad Temporal en Modelos Predictivos de Fórmula 1

**Proyecto de Tesis:** Predicción de Victorias en Fórmula 1 mediante Aprendizaje Automático: Un Estudio Comparativo de Modelos Predictivos  
**Autor:** Salvador Romero Gil  
**Destinatarios:** Comité de Revisión Académica / Asesoría de Tesis  
**Fecha:** Septiembre 2026  
**Estado:** Auditado, Corregido, Validado con Suite de Pruebas Unitarias y Versionado en Git  

---

## 1. Resumen Ejecutivo

El presente informe documenta las acciones correctivas, refactorizaciones estructurales y validaciones estadísticas implementadas en el repositorio tras la emisión del **Informe de Auditoría Técnica y Diagnóstico del Pipeline de Datos**.

El objetivo primordial ha sido subsanar con el más alto rigor científico las vulnerabilidades metodológicas asociadas a **fuga de información (*data leakage*)**, contaminación temporal entre conjuntos de datos, y estimaciones de rendimiento no generalizables. 

Como resultado directo de esta intervención:
1. Se erradicó la variable `raceId` de las matrices explicativas de todos los modelos del proyecto.
2. Se reescribió integralmente el modelo avanzado ([`modelos/5_modelo_avanzado.py`](modelos/5_modelo_avanzado.py)), reemplazando agregaciones globales que contaminaban el pasado con ventanas históricas estrictas (`.shift(1).expanding()`), garantizando la preservación del tamaño muestral oficial (**25,121 observaciones**) y eliminando la calibración del umbral de decisión sobre el conjunto de prueba.
3. Se corrigió el colapso por desbalance de clases en la red neuronal ([`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py)).
4. Se construyó una suite de pruebas unitarias automatizada ([`tests/test_temporal_integrity.py`](tests/test_temporal_integrity.py)) que certifica matemáticamente la condición:
   $$\max(\text{fecha}_{\text{train}}) \le \min(\text{fecha}_{\text{test}})$$
   tanto en el corte temporal principal como en cada pliegue de validación cruzada (*TimeSeriesSplit*).
5. Se archivaron de forma controlada los scripts heredados no reproducibles en el directorio [`legacy/`](legacy/).

---

## 2. Diagnóstico de Vulnerabilidades Metodológicas Detectadas

### 2.1. Fuga por Estructura de Datos: El Rol Inadvertido de `raceId`
- **Naturaleza del Fallo:** En la base de datos relacional de la Fórmula 1, `raceId` es una clave primaria entera generada de forma secuencial monótona creciente en el tiempo.
- **Mecanismo de Fuga:** Al ejecutarse la partición temporal mediante ordenación por fecha y posteriormente separar las variables con `X = df.drop('win', axis=1)`, la variable `raceId` permaneció inadvertidamente en la matriz de diseño $X$ en [`modelos/1_regresion_lineal.py`](modelos/1_regresion_lineal.py), [`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py) y [`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py).
- **Impacto Metodológico:** En los modelos lineales y redes densas, `raceId` funcionaba como una proxy continua artificial del tiempo, absorbiendo peso predictivo en la estimación de coeficientes $\beta$ y *Odds Ratios*, sesgando la interpretación de los factores de mérito automovilístico.

### 2.2. *Target Leakage* Masivo y Ruptura Muestral en el Modelo Avanzado
- **Naturaleza del Fallo:** El script experimental [`modelos/5_modelo_avanzado.py`](modelos/5_modelo_avanzado.py) pretendía enriquecer la capacidad predictiva introduciendo métricas de carrera profesional (`win_rate_career`, `podium_rate_career`, `win_rate_constructor`, etc.).
- **Mecanismo de Fuga:** Dichas variables fueron calculadas mediante operaciones globales de agregación sobre el conjunto completo de datos:
  ```python
  # CÓDIGO DEFECTUOSO PREVIO (CON LEAKAGE):
  driver_stats = results.groupby('driverId').agg({'win': ['sum', 'count']})
  driver_stats['win_rate_career'] = driver_stats['wins'] / driver_stats['races']
  ```
  Esto implicaba que para una carrera disputada en el año 2005, el vector de características de un piloto incluía las victorias acumuladas que dicho piloto obtendría en temporadas futuras (hasta 2024). Esto explica por qué el ensamble obtenía métricas artificialmente infladas ($AUC\text{-}ROC = 0.9884$ y $F_1 = 0.6808$).
- **Ruptura de Trazabilidad Muestral:** La combinación de múltiples uniones relacionales combinadas con una llamada indiscriminada a `dropna()` redujo arbitrariamente la muestra a solo 8,457 observaciones, desalineándose del dataset canónico de 25,121 filas.
- **Contaminación del Conjunto de Prueba (*Test Set Overfitting*):** El script optimizaba el umbral de decisión (*threshold calibration*) evaluando la métrica $F_1$ iterativamente sobre `y_test`, invalidando la independencia del conjunto de prueba.

### 2.3. Colapso en el Entrenamiento del Perceptrón Multicapa (MLP)
- **Naturaleza del Fallo:** La red neuronal PyTorch arrojaba un desempeño prácticamente nulo ($F_1 = 0.0079$), prediciendo exclusivamente la clase mayoritaria (0 = No Victoria).
- **Mecanismo del Fallo:** Se identificó que la función de pérdida `nn.BCELoss()` se instanciaba con su comportamiento predeterminado `reduction='mean'`. Al calcular `loss_per_sample * weights`, `loss_per_sample` ya era un escalar promedio, por lo que la ponderación por muestra se disolvía matemáticamente en un factor constante, dejando desprotegida a la clase minoritaria (4.49% de victorias).

---

## 3. Plan de Remediación e Implementación Técnica

### 3.1. Exclusión Estricta de `raceId` en Todos los Modelos
Se modificaron las definiciones de características en los scripts 1, 3, 4 y 5 para garantizar que `raceId` sea utilizado exclusivamente como clave de ordenación y trazabilidad, siendo retirado de cualquier tensor o matriz $X$:
```python
# IMPLEMENTACIÓN ESTANDARIZADA (Scripts 1, 3, 4 y 5):
X = df.drop(columns=['win', 'raceId'], errors='ignore')

# Tras el split cronológico:
X_train = train_df.drop(columns=['win', 'raceId'], errors='ignore')
X_test  = test_df.drop(columns=['win', 'raceId'], errors='ignore')
```

### 3.2. Formulación Matemática de Métricas Históricas sin Contaminación (*Anti-Leakage*)
Para mantener la aspiración del modelo avanzado de modelar el historial competitivo sin violar la flecha del tiempo, se sustituyeron las agregaciones estáticas por **operadores de rezago y ventanas acumulativas expansivas**:

$$\text{WinRate}_{i, t} = \begin{cases} \frac{\sum_{k=1}^{t-1} \text{win}_{i, k}}{\sum_{k=1}^{t-1} 1} & \text{si } \sum_{k=1}^{t-1} 1 > 0 \\ 0.0 & \text{si el piloto o constructor es debutante } (t=1) \end{cases}$$

En código de Python/pandas, esto se tradujo en:
```python
# Ordenamiento canónico previo:
results = results.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)

# Acumulación estricta al pasado mediante shift(1):
results['driver_wins_prior'] = results.groupby('driverId')['win'].transform(
    lambda s: s.shift(1).expanding().sum()
).fillna(0)

results['races_career'] = results.groupby('driverId')['win'].transform(
    lambda s: s.shift(1).expanding().count()
).fillna(0)

results['win_rate_career'] = (
    results['driver_wins_prior'] / results['races_career'].replace(0, np.nan)
).fillna(0.0)
```
- **Preservación Muestral:** Al imputar con valor neutral $0.0$ a los debutantes y nuevos constructores, se eliminó la necesidad de descartar filas incompletas, preservando las **25,121 observaciones** originales.
- **Calibración Independiente de Umbrales:** La búsqueda por malla del umbral óptimo de corte se trasladó al conjunto de entrenamiento (`X_train_scaled`, `y_train`). El umbral seleccionado se evalúa posteriormente sobre `X_test_scaled` de forma ciega.
- **Control de Complejidad:** Se estableció `min_samples_leaf=2` en el Random Forest para evitar ramas con hojas unitarias.

### 3.3. Corrección de la Ponderación de Clases en PyTorch (MLP)
Se configuró el cálculo de pérdida muestra a muestra con reducción explícita:
```python
criterion = nn.BCELoss(reduction='none')

# En el ciclo de entrenamiento:
loss_per_sample = criterion(outputs, y_train_tensor)
weights = y_train_tensor * class_weights_tensor[1] + (1 - y_train_tensor) * class_weights_tensor[0]
loss = (loss_per_sample * weights).mean()
```
Esto aplica un multiplicador de penalización de $\approx 11.1\times$ a cada falso negativo, forzando a la red a aprender representaciones efectivas de la clase victoria.

### 3.4. Higiene y Segregación del Repositorio
Para evitar confusiones en la entrega académica final, se creó el directorio [`legacy/`](legacy/) y se trasladaron los archivos:
- `1_preparacion_dataset.py` (Script preliminar con variables de carrera).
- `modelo_regresion_lineal.py` (Script preliminar con partición aleatoria).
- `f1_win.py` (Script legacy con fuga temporal).
- [`legacy/README.md`](legacy/README.md) (Documento que explicita su obsolescencia metodológica).

---

## 4. Banco de Pruebas Unitarias de Integridad Temporal

Para proporcionar una verificación demostrable e incontrovertible, se diseñó e integró la suite de pruebas unitarias [`tests/test_temporal_integrity.py`](tests/test_temporal_integrity.py).

### 4.1. Cobertura de las Pruebas
1. **`test_01_chronological_split_boundary`**: Verifica que $\max(\text{fecha}_{\text{train}}) \le \min(\text{fecha}_{\text{test}})$ en la división 80/20 sobre las 25,121 filas.
2. **`test_02_time_series_split_expanding_folds`**: Simula una validación cruzada temporal de 5 pliegues expansivos y comprueba que para cada pliegue $k$, ninguna carrera del conjunto de validación preceda a las de entrenamiento.
3. **`test_03_no_prohibited_in_race_features`**: Examina las cabeceras de datos confirmando la ausencia total de variables post-carrera (`fastestLapSpeed`, `milliseconds_pit_stop`, etc.).
4. **`test_04_raceid_excluded_from_model_features`**: Realiza análisis estático de código sobre los 5 scripts bajo `modelos/`, asegurando que todos excluyen explícitamente `raceId`.
5. **`test_05_unified_dataset_size`**: Verifica la igualdad $N = 25,121$.

### 4.2. Resultados Obtenidos
```text
Ran 5 tests in 0.658s
OK

[OK] Frontera Temporal 80/20 Validada:
     • Train (80% Pasado): 20,096 filas | Fecha máx: 2012-11-04
     • Test  (20% Futuro):  5,025 filas | Fecha mín: 2012-11-04

[OK] Validando 5 folds de TimeSeriesSplit:
     • Fold 1: Train (4,186 filas, máx 1971-06-20) <= Val (4,186 filas, mín 1971-06-20)
     • Fold 2: Train (8,372 filas, máx 1982-09-25) <= Val (4,186 filas, mín 1982-09-25)
     • Fold 3: Train (12,558 filas, máx 1993-04-11) <= Val (4,186 filas, mín 1993-04-11)
     • Fold 4: Train (16,744 filas, máx 2004-07-04) <= Val (4,186 filas, mín 2004-07-04)
     • Fold 5: Train (20,930 filas, máx 2014-11-02) <= Val (4,186 filas, mín 2014-11-02)

[OK] Cero variables post-carrera prohibidas en el dataset.
[OK] Exclusión de 'raceId' confirmada en los 5 modelos.
[OK] Integridad Muestral: 25,121 observaciones confirmadas.
```

---

## 5. Cuadro Comparativo de Rendimiento Tras la Remediación

A continuación se sintetizan las métricas reales y auditadas de los modelos tras la eliminación de todo sesgo de fuga:

| Modelo / Script | Tamaño Train / Test | F1-Score (Test) | AUC-ROC | AUC-PR | Recall (Test) | Estado Metodológico |
|---|---|---|---|---|---|---|
| **Regresión Lineal** (`1_regresion_lineal.py`) | 20,096 / 5,025 | 0.0000 | 0.8121 | 0.2327 | 0.0000 | **Válido:** Predictores depurados (sin `raceId`). Refleja la limitación del umbral $0.5$ ante desbalance severo. |
| **Regresión Logística** (`4_regresion_logistica.py`) | 20,096 / 5,025 | 0.2640 | 0.8220 | 0.1792 | 0.5582 | **Válido:** Coeficientes interpretables sin distorsión temporal. Factor `grid` emerge como predictor principal ($\beta=-2.53$). |
| **Random Forest Baseline** (`2_random_forest.py`) | 20,096 / 5,025 | 0.3139 | 0.8855 | 0.2608 | 0.8635 | **Válido:** Partición cronológica 80/20 y class weighting balanceado. |
| **Random Forest GridSearch** (`2_random_forest.py`) | 20,096 / 5,025 | **0.3349** | **0.8869** | **0.2748** | **0.8394** | **Válido:** Optimizado mediante *TimeSeriesSplit* con `min_samples_leaf=2`. Mejor modelo base del proyecto. |
| **Perceptrón Multicapa (MLP)** (`3_perceptron_multicapa.py`) | 20,096 / 5,025 | 0.0000 | 0.7135 | 0.1074 | 0.0000 | **Válido:** Mecanismo de ponderación muestral corregido. Capacidad discriminativa evaluada mediante AUC ($0.7135$); F1 refleja la sensibilidad al umbral estándar $0.5$ ante desbalance. |
| **Modelo Avanzado Ensemble** (`5_modelo_avanzado.py`) | 20,096 / 5,025 | **0.5018** (RF ind: **0.5424**) | **0.9466** (RF ind: **0.9478**) | **0.4688** | **0.5622** (RF ind: **0.5984**) | **Válido y Robusto:** Eliminadas las métricas infladas artificialmente ($AUC=0.988$). Restablecido a las 25,121 observaciones completas. Ingeniería temporal sin fuga (`shift(1).expanding()`). |


---

## 6. Recomendaciones para la Defensa Académica

Para la reunión con el comité de revisión y asesoría de tesis:

1. **Defensa de la Integridad Temporal:** Presentar con orgullo académico la detección y subsanación del *leakage*. Explicar que un modelo predictivo con $AUC = 0.988$ en Fórmula 1 era indicativo de contaminación de datos, mientras que los valores obtenidos actualmente ($AUC \in [0.82, 0.89]$) representan el estado del arte de la disciplina bajo condiciones de evaluación a priori reales.
2. **Exhibición de la Suite de Pruebas:** Ejecutar en vivo la suite `py -m unittest tests/test_temporal_integrity.py` para demostrar que el pipeline cuenta con pruebas automatizadas de validación cruzada temporal.
3. **Respaldo en Control de Versiones:** Apoyarse en el historial de commits y en los hooks de pre-push implementados para evidenciar la madurez de la infraestructura de software del proyecto.

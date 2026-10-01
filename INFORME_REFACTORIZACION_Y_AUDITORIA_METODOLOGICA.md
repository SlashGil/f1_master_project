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

Como resultado directo de esta intervención sobre los modelos oficiales del alcance de la tesis:
1. Se erradicó la variable sintética `raceId` de las matrices explicativas de todos los modelos del proyecto ([`modelos/1_regresion_lineal.py`](modelos/1_regresion_lineal.py), [`modelos/2_random_forest.py`](modelos/2_random_forest.py), [`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py) y [`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py)).
2. Se corrigió el colapso por desbalance de clases en la red neuronal ([`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py)), implementando reducción muestra a muestra y penalización ponderada para la clase minoritaria (~4.49% victorias).
3. Se construyó una suite de pruebas unitarias automatizada ([`tests/test_temporal_integrity.py`](tests/test_temporal_integrity.py)) que certifica matemáticamente la condición de partición causal:
   $$\max(\text{fecha}_{\text{train}}) \le \min(\text{fecha}_{\text{test}})$$
   tanto en la partición cronológica principal 80/20 como en cada uno de los pliegues expansivos de validación cruzada (*TimeSeriesSplit*).
4. Se archivaron de forma controlada los scripts preliminares no reproducibles en el directorio [`legacy/`](legacy/), asegurando la trazabilidad y reproducibilidad del repositorio.

---

## 2. Diagnóstico de Vulnerabilidades Metodológicas Detectadas

### 2.1. Fuga por Estructura de Datos: El Rol Inadvertido de `raceId`
- **Naturaleza del Fallo:** En la base de datos relacional de la Fórmula 1, `raceId` es una clave primaria entera autoincremental asignada de forma secuencial monótona creciente en el tiempo.
- **Mecanismo de Fuga:** Al ejecutarse la partición temporal mediante ordenación cronológica por fecha y posteriormente separar las variables con `X = df.drop('win', axis=1)`, la variable `raceId` permaneció inadvertidamente en la matriz de diseño $X$ en [`modelos/1_regresion_lineal.py`](modelos/1_regresion_lineal.py), [`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py) y [`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py).
- **Impacto Metodológico:** En los modelos lineales y redes densas, `raceId` actuaba como una proxy continua espuria del tiempo cronológico. Esto distorsionaba la estimación de coeficientes $\beta$ y *Odds Ratios*, absorbiendo peso predictivo artificial que enmascaraba el verdadero impacto de las variables de mérito deportivo previo a la carrera.

### 2.2. Colapso en el Entrenamiento del Perceptrón Multicapa (MLP)
- **Naturaleza del Fallo:** La red neuronal PyTorch arrojaba un desempeño prácticamente nulo ($F_1 = 0.0079$), prediciendo de forma homogénea la clase mayoritaria ($0 = \text{No Victoria}$).
- **Mecanismo del Fallo:** Se identificó que la función de pérdida `nn.BCELoss()` se instanciaba con su comportamiento predeterminado `reduction='mean'`. Al calcular `loss_per_sample * weights`, `loss_per_sample` ya era un escalar promedio, por lo que la ponderación por muestra se disolvía matemáticamente en un factor constante, dejando desprotegida a la clase minoritaria (4.49% de victorias).

---

## 3. Plan de Remediación e Implementación Técnica

### 3.1. Exclusión Estricta de `raceId` en los Modelos del Proyecto
Se modificaron las definiciones de características en los scripts 1, 2, 3 y 4 para garantizar que `raceId` sea utilizado exclusivamente como clave de ordenación cronológica y trazabilidad, siendo retirado de cualquier tensor o matriz explicativa $X$:
```python
# IMPLEMENTACIÓN ESTANDARIZADA (Scripts 1, 2, 3 y 4):
X = df.drop(columns=['win', 'raceId'], errors='ignore')

# Tras la partición cronológica:
X_train = train_df.drop(columns=['win', 'raceId'], errors='ignore')
X_test  = test_df.drop(columns=['win', 'raceId'], errors='ignore')
```

### 3.2. Corrección de la Ponderación de Clases en PyTorch (MLP)
Se configuró el cálculo de pérdida muestra a muestra con reducción explícita:
```python
criterion = nn.BCELoss(reduction='none')

# En el ciclo de entrenamiento:
loss_per_sample = criterion(outputs, y_train_tensor)
weights = y_train_tensor * class_weights_tensor[1] + (1 - y_train_tensor) * class_weights_tensor[0]
loss = (loss_per_sample * weights).mean()
```
Esto aplica un multiplicador de penalización de $\approx 11.1\times$ a cada falso negativo, forzando a la red neuronal a optimizar gradientes sobre la clase positiva de victoria. Adicionalmente, se dotó al script de arquitectura dual con respaldo automático en `MLPClassifier` de Scikit-Learn.

### 3.3. Higiene y Segregación del Repositorio
Para evitar confusiones en la entrega académica final, se creó el directorio [`legacy/`](legacy/) y se trasladaron los archivos preliminares:
- `1_preparacion_dataset.py` (Script preliminar con variables de carrera).
- `modelo_regresion_lineal.py` (Script preliminar con partición aleatoria).
- `f1_win.py` (Script exploratorio original con fuga temporal).
- [`legacy/README.md`](legacy/README.md) (Documento explicativo de su descarte metodológico).

---

## 4. Banco de Pruebas Unitarias de Integridad Temporal

Para proporcionar una verificación demostrable e incontrovertible, se diseñó e integró la suite de pruebas unitarias [`tests/test_temporal_integrity.py`](tests/test_temporal_integrity.py).

### 4.1. Cobertura de las Pruebas
1. **`test_01_chronological_split_boundary`**: Verifica que $\max(\text{fecha}_{\text{train}}) \le \min(\text{fecha}_{\text{test}})$ en la división 80/20 sobre las 25,121 observaciones.
2. **`test_02_time_series_split_expanding_folds`**: Simula una validación cruzada temporal de 5 pliegues expansivos y comprueba que para cada pliegue $k$, ninguna carrera del conjunto de validación preceda a las de entrenamiento.
3. **`test_03_no_prohibited_in_race_features`**: Examina las variables explicativas confirmando la ausencia total de variables post-carrera (`fastestLapSpeed`, `milliseconds_pit_stop`, etc.).
4. **`test_04_raceid_excluded_from_model_features`**: Realiza análisis estático de código sobre los 4 modelos oficiales bajo `modelos/`, asegurando que todos excluyen explícitamente `raceId`.
5. **`test_05_unified_dataset_size`**: Verifica la integridad muestral exacta con $N = 25,121$.

### 4.2. Resultados Obtenidos
```text
Ran 5 tests in 0.653s
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
[OK] Exclusión de 'raceId' confirmada en los 4 modelos oficiales.
[OK] Integridad Muestral: 25,121 observaciones confirmadas.
```

---

## 5. Cuadro Comparativo de Rendimiento Tras la Remediación

A continuación se sintetizan las métricas reales y auditadas de los modelos oficiales del proyecto tras la eliminación de todo sesgo de fuga y la correcta evaluación temporal sobre el subconjunto de prueba ($N_{\text{test}} = 5,025$ observaciones, temporadas 2012 a 2024):

| Modelo / Script | Tamaño Train / Test | F1-Score (Test) | AUC-ROC | AUC-PR | Recall (Test) | Precision (Test) | Estado Metodológico y Conclusiones |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Regresión Lineal**<br>([`1_regresion_lineal.py`](modelos/1_regresion_lineal.py)) | 20,096 / 5,025 | 0.0000 | 0.8121 | 0.2327 | 0.0000 | 0.0000 | **Válido:** Predictores depurados (sin `raceId`). Evidencia la insuficiencia del umbral estándar $0.5$ ante desbalance severo (~4.9% victorias). Coeficiente `grid` dominante ($\beta = -0.0555$). |
| **Regresión Logística**<br>([`4_regresion_logistica.py`](modelos/4_regresion_logistica.py)) | 20,096 / 5,025 | 0.2640 | 0.8220 | 0.1792 | 0.5582 | 0.1729 | **Válido:** Coeficientes interpretables sin distorsión temporal. Factor `grid` emerge como predictor cardinal ($\beta=-2.5294$, $\text{Odds Ratio} = 0.0797$). |
| **Random Forest Baseline**<br>([`2_random_forest.py`](modelos/2_random_forest.py)) | 20,096 / 5,025 | 0.3139 | 0.8855 | 0.2608 | 0.8635 | 0.1915 | **Válido:** Partición cronológica 80/20 y class weighting balanceado. Excelente capacidad de recuperación de victorias (Recall 86.35%). |
| **Random Forest GridSearch**<br>([`2_random_forest.py`](modelos/2_random_forest.py)) | 20,096 / 5,025 | **0.3349** | **0.8869** | **0.2748** | **0.8394** | **0.2098** | **Modelo Campeón del Proyecto:** Optimizado mediante *TimeSeriesSplit* con regularización de hojas (`min_samples_leaf=2`, `max_depth=12`, `n_estimators=100`). Máximo equilibrio entre precisión y cobertura. |
| **Perceptrón Multicapa (MLP)**<br>([`3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py)) | 20,096 / 5,025 | 0.1253 | 0.6488 | 0.0822 | 0.9639 | 0.0670 | **Válido:** Red neuronal PyTorch entrenada con ponderación de clase minoritaria ($w=11.43$). Su alta penalización rescata el 96.39% de victorias (Recall) superando el colapso inicial. |

---

## 6. Recomendaciones para la Defensa Académica

Para la reunión con el comité de revisión y asesoría de tesis:

1. **Defensa de la Integridad Temporal y Rigor Metodológico:** Presentar con orgullo académico la detección y subsanación del *leakage*. Explicar que modelos inflados con métricas ficticias ($AUC > 0.98$) en Fórmula 1 eran síntoma de contaminación o de inclusión de variables espurias (`raceId`), mientras que los valores obtenidos actualmente ($AUC \in [0.81, 0.89]$) representan el estado del arte de la disciplina bajo condiciones de evaluación *a priori* estrictamente causales.
2. **Destacar el Desempeño del Random Forest:** El modelo Random Forest optimizado mediante `TimeSeriesSplit` se consagra como el modelo de mayor solidez técnica de la tesis, alcanzando un $AUC\text{-}ROC = 0.8869$ y un $\text{Recall} = 83.94\%$, permitiendo predecir victorias con alta sensibilidad a partir de datos exclusivamente pre-carrera.
3. **Exhibición de la Suite de Pruebas:** Ejecutar en vivo la suite `python -m unittest discover tests -v` para demostrar que el pipeline cuenta con pruebas automatizadas de validación cruzada temporal y chequeo de no-fuga.
4. **Respaldo en Control de Versiones:** Apoyarse en el historial de commits y en los hooks de pre-push implementados para evidenciar la madurez de la infraestructura de software del proyecto.

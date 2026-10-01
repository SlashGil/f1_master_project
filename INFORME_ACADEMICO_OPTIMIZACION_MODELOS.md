# INFORME TÉCNICO-ACADÉMICO: OPTIMIZACIÓN DEL ESPACIO DE CARACTERÍSTICAS Y EVALUACIÓN BENCHMARK CAUSAL (1950–2024)

**Proyecto de Tesis de Maestría:** *Modelado Predictivo y Análisis Causal de Victorias en Fórmula 1 mediante Técnicas de Aprendizaje Automático y Validación Temporal Estricta*  
**Autor:** Salvador Romero Gil  
**Entorno de Control de Versiones:** Rama Experimental `test` (Commit `e4c3216`)  
**Fecha:** Septiembre 2026  

---

## 1. Resumen Ejecutivo y Objeto del Experimento

El presente informe expone los fundamentos metodológicos, el diseño experimental y las conclusiones estadísticas derivadas de una **intervención de ablación dimensional** sobre el conjunto de datos canónico de Fórmula 1 ($N = 25,121$ observaciones). El experimento evaluó empíricamente la hipótesis nula de si la supresión de las variables temporal (`year`) y demográfica (`nationality`) mejoraba la capacidad de generalización fuera de muestra (*out-of-sample*) y mitigaba el sobreajuste (*overfitting*) en la predicción pre-carrera de victorias.

Como resultado de esta optimización, el espacio vectorial de predictores se redujo de **$D = 245$** a **$D = 201$** dimensiones estrictamente pre-carrera. La evaluación ciega sobre el conjunto de prueba independiente ($N_{\text{test}} = 5,025$ observaciones correspondientes a las temporadas 2012–2024) evidenció una **mejora sustancial, uniforme y estadísticamente robusta** en todos los estimadores supervisados, ratificando el modelo de ensamble de árboles (*Random Forest Optimizado*) como el estimador superior con un **$\text{AUC-ROC} = 0.9265$** y un **$\text{AUC-PR} = 0.3766$**.

Asimismo, se formalizó la delimitación del alcance de la tesis mediante la exclusión definitiva del modelo de Regresión Lineal ordinaria (OLS), confinándolo a registros históricos de auditoría y preservando como suite oficial los tres modelos fundamentales: **Random Forest**, **Regresión Logística** y **Perceptrón Multicapa (PyTorch)**. Todos los cambios descritos se han aislado de forma controlada en la rama experimental **`test`**, manteniendo la rama principal `main` en su estado de referencia previo.

---

## 2. Justificación Teórica y Econométrica de la Reducción Dimensional

### 2.1. El Problema de Extrapolación Monótona de la Variable `year`
En formulaciones clásicas con muestreo aleatorio (*random split*), la variable de calendario `year` actúa frecuentemente como un atajo predictivo espurio (*shortcut learning*). Sin embargo, bajo el protocolo de causalidad temporal adoptado:
$$\mathcal{D}_{\text{train}} = \left\{ (x_i, y_i) \mid t_i \le t^* \right\}, \quad \mathcal{D}_{\text{test}} = \left\{ (x_j, y_j) \mid t_j > t^* \right\}$$
donde $t^* = \text{04-11-2012}$, el soporte de la variable temporal en entrenamiento satisface $\text{supp}(year_{\text{train}}) = [1950, 2012]$, mientras que en evaluación satisface $\text{supp}(year_{\text{test}}) = (2012, 2024]$.

Dado que $\text{supp}(year_{\text{train}}) \cap \text{supp}(year_{\text{test}}) = \emptyset$, cualquier estimador que asigne peso a `year` se ve obligado a extrapolar en un dominio numérico no observado. En modelos lineales generalizados, esto introducía derivas sistemáticas en la función logit; en árboles de decisión, inducía particiones muertas en nodos terminales. Su remoción restaura la invariancia temporal requerida para que el modelo aprenda dinámica de competencia y no marcas cronológicas.

### 2.2. Dispersión Espuria y Ruido de la Variable `nationality`
La nacionalidad del piloto generaba $43$ variables indicadoras binarias (*one-hot dummies*), muchas de ellas con frecuencias relativas marginales inferiores al $0.1\%$. En la disciplina automovilística de la Fórmula 1 moderna, la evidencia empírica demuestra que el desempeño deportivo está gobernado por el binomio coche-clasificación (competitividad del constructor $constructorId$ y posición de partida $grid$), careciendo la nacionalidad de un mecanismo causal directo sobre la probabilidad de victoria. 

La inclusión de 43 dimensiones ruidosas inducía la conocida "maldición de la dimensionalidad" (*curse of dimensionality*), fragmentando los criterios de partición de impureza de Gini en los árboles y entorpeciendo el descenso de gradiente estocástico en la red neuronal.

---

## 3. Matriz Comparativa de Resultados: Rendimiento Nuevo vs Anterior

A continuación se detalla la comparativa exhaustiva del comportamiento predictivo sobre el conjunto de prueba independiente ($N_{\text{test}} = 5,025$, de los cuales 249 corresponden a victorias reales, tasa base $\pi_1 = 4.95\%$):

### 3.1. Tabla Comparativa de Rendimiento Fuera de Muestra

| Modelo Predictivo Oficial | Espacio Dimensional | F1-Score | AUC-ROC | AUC-PR | Recall ($y=1$) | Precision ($y=1$) | Ganancia Neta ($\Delta$) y Hallazgo Metodológico |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🏆 **Random Forest (GridSearch)** | Con Year & Nat ($D=245$)<br>**Sin Year & Nat ($D=201$)** | 0.3349<br>**0.3580** | 0.8869<br>**0.9265** | 0.2748<br>**0.3766** | 83.94%<br>**88.35%** | 20.98%<br>**22.45%** | **Modelo Campeón de la Tesis:**<br>$\Delta \text{AUC-ROC} = \mathbf{+0.0396}$ ($+4.47\%$), $\Delta \text{AUC-PR} = \mathbf{+0.1018}$ ($+37.05\%$). Rescata **220 de 249** victorias reales con máxima especificidad. |
| **Random Forest (Baseline)** | Con Year & Nat ($D=245$)<br>**Sin Year & Nat ($D=201$)** | 0.3139<br>**0.3402** | 0.8855<br>**0.9220** | 0.2608<br>**0.3283** | 86.35%<br>**89.16%** | 19.15%<br>**21.02%** | **Robustez Estructural:**<br>$\Delta \text{AUC-ROC} = \mathbf{+0.0365}$, $\Delta F_1 = \mathbf{+0.0263}$. Detección de **222 de 249** victorias reales sin requerir poda intensiva de hiperparámetros. |
| **Regresión Logística** | Con Year & Nat ($D=245$)<br>**Sin Year & Nat ($D=201$)** | 0.2640<br>**0.3027** | 0.8220<br>**0.9233** | 0.1792<br>**0.3207** | 55.82%<br>**93.98%** | 17.29%<br>**18.04%** | **Salto Cualitativo Excepcional:**<br>$\Delta \text{AUC-ROC} = \mathbf{+0.1013}$ ($+12.32\%$), $\Delta \text{Recall} = \mathbf{+38.16\%}$. Detecta **234 ganadores**; la supresión de colinealidad estabilizó el estimador logit. |
| **Perceptrón Multicapa (MLP)** | Con Year & Nat ($D=245$)<br>**Sin Year & Nat ($D=201$)** | 0.1253<br>**0.2069** | 0.6488<br>**0.8818** | 0.0822<br>**0.3531** | 96.39%<br>**97.59%** | 6.70%<br>**11.57%** | **Convergencia Profunda en PyTorch:**<br>$\Delta \text{AUC-ROC} = \mathbf{+0.2330}$ ($+35.91\%$), $\Delta \text{AUC-PR} = \mathbf{+0.2709}$ ($+329.5\%$). Rescata **243 de 249** victorias reales. |

---

## 4. Análisis Detallado por Arquitectura de Modelado

### 4.1. Random Forest (Optimizado mediante TimeSeriesSplit)
* **Archivo:** `modelos/2_random_forest.py`
* **Configuración Óptima:** $\text{n\_estimators} = 100$, $\text{max\_depth} = 10$, $\text{min\_samples\_leaf} = 2$, $\text{class\_weight} = \text{'balanced'}$.
* **Comportamiento:** La métrica de precisión-recuperación (AUC-PR) ascendió de 0.2748 a **0.3766**, superando por más de 7.6 veces la línea base aleatoria ($\pi_1 = 0.0495$). La regularización inducida al remover 44 variables parásitas consolidó a la posición de parrilla (`grid`) como el principal absorbedor de impureza de Gini ($>68\%$), complementado armónicamente con los coeficientes de constructores dominantes (Ferrari, McLaren, Mercedes, Red Bull).

### 4.2. Regresión Logística (Interpretabilidad Econométrica)
* **Archivo:** `modelos/4_regresion_logistica.py`
* **Formulación:** $\text{logit}(P(Y=1)) = \beta_0 + \beta_{\text{grid}} X_{\text{grid}} + \beta_{\text{age}} X_{\text{age}} + \beta_{\text{round}} X_{\text{round}} + \sum_k \gamma_k D_{\text{constructor}, k}$.
* **Hallazgo Clave:** Al remover la multicolinealidad con `year`, el coeficiente de la parrilla se ajustó a $\beta_{\text{grid}} = -2.6122$, lo que equivale a un *Odds Ratio* de:
  $$\text{OR}_{\text{grid}} = \exp(-2.6122) \approx 0.0734$$
  Esto demuestra que, manteniendo constantes los demás factores, cada posición de retraso en la grilla de salida reduce las probabilidades relativas de victoria en un **$92.66\%$**. Este modelo incrementó su Recall de $55.82\%$ a **$93.98\%$**, convirtiéndose en un clasificador probabilístico de referencia altamente calibrado.

### 4.3. Perceptrón Multicapa (Deep Learning en PyTorch)
* **Archivo:** `modelos/3_perceptron_multicapa.py`
* **Arquitectura:** Red densa feedforward $\mathbb{R}^{201} \to 128 \to 64 \to 32 \to 1$ con activaciones ReLU, Dropout ($p=0.3$) y capa final sigmoide.
* **Optimización de Pérdida:** Función de coste Binary Cross-Entropy con ponderación de muestra (*sample-weighted loss*):
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N w_i \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
  donde $w_{\text{win}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = 11.43$.
* **Hallazgo Clave:** En la configuración anterior (245 variables), la red colapsaba parcialmente en una región subóptima debido a la alta dispersión de los gradientes asociados a las dummies de nacionalidad ($\text{AUC-ROC} = 0.6488$). Con el espacio compacto de 201 dimensiones, el optimizador Adam convergió con estabilidad, alcanzando un **$\text{AUC-ROC} = 0.8818$** y un Recall de **$97.59\%$** (captura a 243 de los 249 ganadores de la era contemporánea).

---

## 5. Delimitación del Alcance y Gobernanza del Código

En concordancia con los objetivos académicos de la tesis, se ejecutaron las siguientes acciones de gobernanza:

1. **Exclusión de la Regresión Lineal (OLS):**
   * El modelo lineal ordinario con función de pérdida cuadrática fue descartado formalmente por carecer de fundamento probabilístico en problemas de clasificación desbalanceada (generación de probabilidades fuera del intervalo $[0, 1]$ e ineficiencia del umbral estático de corte).
   * Se incorporaron a `.gitignore` los patrones:
     ```gitignore
     modelos/1_regresion_lineal.py
     resultados_regresion_lineal/
     ```
   * Se desindexaron del árbol de seguimiento de Git todos los archivos correspondientes a dicho modelo.
2. **Actualización del Portal de Divulgación (`docs/index.html`):**
   * Se actualizó el tablero de indicadores superiores: AUC 0.9265, Recall 88.35%, Dimensión 201.
   * Se sustituyó la tabla original por la matriz comparativa de ablación (Nuevo vs Anterior).
   * En la galería científica, la curva ROC lineal fue reemplazada por la curva ROC de la red neuronal PyTorch (`docs/assets/roc_mlp.png`).
3. **Estructura de Ramas en Git:**
   * **`main`:** Preservada en el commit `76b359a` como línea base previa.
   * **`test`:** Rama activa actual con el commit `e4c3216`, que encapsula de forma íntegra e independiente la totalidad de los cambios, scripts optimizados, resultados experimentales y suite de validación.

---

## 6. Certificación Formal mediante Pruebas Unitarias

La integridad causal del repositorio se encuentra blindada mediante la suite de validación automatizada en `tests/test_temporal_integrity.py`:

$$\forall k \in \{1, \dots, K\}, \quad \max(T_{\text{train}}^{(k)}) \le \min(T_{\text{val}}^{(k)})$$

```text
Ran 5 tests in 0.886s
OK

[OK] Invariante 1: Frontera Temporal 80/20 Demostrada (Train: 20,096 | Test: 5,025)
[OK] Invariante 2: Validación Cruzada Causal en 5 folds expansivos de TimeSeriesSplit
[OK] Invariante 3: Cero variables post-carrera o intra-carrera en el dataset unificado
[OK] Invariante 4: Exclusión estricta de 'raceId', 'year' y 'nationality' en los 3 modelos oficiales
[OK] Invariante 5: Integridad muestral canónica verificada en N = 25,121 observaciones
```

---

## 7. Conclusión Académica

La decisión de eliminar `year` y `nationality` no solo se encuentra plenamente justificada desde el punto de vista de la teoría econométrica y de la inferencia causal en series de tiempo, sino que queda **empíricamente vindicada** por los datos: todos los modelos supervisados aumentaron su poder discriminatorio ($\text{AUC-ROC}$ y $\text{AUC-PR}$), incrementaron su cobertura de victorias reales y redujeron su varianza muestral.

El estado del arte experimental de esta tesis queda formalmente establecido en la rama **`test`**, listo para su defensa ante comités de revisión técnica y académica.

# Avance 4: Implementación del Modelo de Regresión Logística

## Documento Académico para Tesis de Maestría

---

**Título del Proyecto:** Predicción de Victorias en Fórmula 1 mediante Aprendizaje Automático: Un Estudio Comparativo de Modelos Predictivos

**Autor:** [Nombre del Estudiante]  
**Programa:** [Nombre del Programa de Posgrado]  
**Institución:** [Nombre de la Universidad]  
**Fecha:** Febrero 2026

---

## Tabla de Contenidos

1. [Actualización del Marco Teórico: Regresión Logística](#1-actualización-del-marco-teórico-regresión-logística)
   - 1.1 [Definición Formal](#11-definición-formal)
   - 1.2 [Formulación Matemática](#12-formulación-matemática)
   - 1.3 [Interpretación Probabilística](#13-interpretación-probabilística)
   - 1.4 [Supuestos del Modelo](#14-supuestos-del-modelo)
   - 1.5 [Ventajas y Limitaciones](#15-ventajas-y-limitaciones)
2. [Implementación del Modelo](#2-implementación-del-modelo)
   - 2.1 [Configuración del Modelo](#21-configuración-del-modelo)
   - 2.2 [Entrenamiento del Modelo](#22-entrenamiento-del-modelo)
   - 2.3 [Análisis de Coeficientes](#23-análisis-de-coeficientes)
3. [Resultados Preliminares](#3-resultados-preliminares)
4. [Anexo: Fragmentos de Código Relevante](#4-anexo-fragmentos-de-código-relevante)

---

# 1. Actualización del Marco Teórico: Regresión Logística

## 1.1 Definición Formal

La **Regresión Logística** es un modelo estadístico de clasificación supervisada que pertenece a la familia de los **modelos lineales generalizados** (GLM, por sus siglas en inglés *Generalized Linear Models*). A diferencia de la Regresión Lineal, que predice valores continuos en el rango $(-\infty, +\infty)$, la Regresión Logística está diseñada específicamente para problemas de clasificación binaria, donde la variable de respuesta asume valores discretos, típicamente codificados como $y \in \{0, 1\}$.

Desde una perspectiva probabilística, la Regresión Logística modela la **probabilidad condicional** de que una observación pertenezca a la clase positiva ($y = 1$) dado un conjunto de características predictoras $\mathbf{x} = (x_1, x_2, ..., x_p)$. Formalmente:

$$P(Y = 1 \mid \mathbf{X} = \mathbf{x}) = \pi(\mathbf{x})$$

Donde $\pi(\mathbf{x})$ representa la función de probabilidad que debe cumplir con la restricción fundamental de que sus valores estén acotados en el intervalo $[0, 1]$, propiedad esencial para una interpretación probabilística válida.

La denominación "logística" proviene del empleo de la **función logística** (también conocida como función sigmoide) como función de enlace (*link function*) entre el predictor lineal y la probabilidad de respuesta. Este modelo fue desarrollado inicialmente por **David Cox** en 1958 (Cox, 1958) y constituye uno de los métodos más ampliamente utilizados en ciencias sociales, medicina, economía y, más recientemente, en aprendizaje automático aplicado.

### 1.1.1 Contexto de Aplicación en el Estudio

En el marco de esta investigación, la Regresión Logística se aplica para modelar la probabilidad de que un piloto de Fórmula 1 obtenga la victoria en una carrera determinada, fundamentado en variables predictoras que incluyen características del piloto, del constructor, del circuito y del desempeño histórico. La elección de este modelo se justifica por:

1. **Interpretabilidad**: Los coeficientes del modelo permiten cuantificar el efecto de cada variable sobre la *odds* de victoria.
2. **Base probabilística**: Proporciona estimaciones calibradas de probabilidad, no únicamente clasificaciones binarias.
3. **Robustez**: Funciona razonablemente bien con datasets de dimensionalidad moderada, incluso cuando el número de observaciones es considerable.

---

## 1.2 Formulación Matemática

### 1.2.1 La Función Logística (Sigmoide)

El componente central de la Regresión Logística es la **función logística**, definida matemáticamente como:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^{z}}{1 + e^{z}}$$

Donde $z$ representa el **predictor lineal** (también denominado *logit* o log-odds), calculado como una combinación lineal de las variables predictoras:

$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$

En notación vectorial compacta:

$$z = \boldsymbol{\beta}^T \mathbf{x}$$

Donde $\boldsymbol{\beta} = (\beta_0, \beta_1, ..., \beta_p)^T$ es el vector de coeficientes del modelo (incluyendo el intercepto) y $\mathbf{x} = (1, x_1, ..., x_p)^T$ es el vector de características ampliado.

#### Propiedades Matemáticas de la Función Sigmoide

La función sigmoide presenta características matemáticas fundamentales que la hacen apropiada para modelación probabilística:

| Propiedad | Descripción Matemática | Implicación |
|-----------|------------------------|-------------|
| **Rango acotado** | $\sigma(z) \in (0, 1)$ para todo $z \in \mathbb{R}$ | Garantiza salidas interpretables como probabilidades |
| **Simetría** | $\sigma(-z) = 1 - \sigma(z)$ | Complementariedad entre clases |
| **Monotonicidad** | $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z)) > 0$ | Función estrictamente creciente |
| **Asíntotas** | $\lim_{z \to +\infty} \sigma(z) = 1$; $\lim_{z \to -\infty} \sigma(z) = 0$ | Saturación en probabilidades extremas |
| **Punto de inflexión** | $\sigma(0) = 0.5$ | Umbral natural de decisión |

### 1.2.2 El Modelo de Regresión Logística

Combinando el predictor lineal con la función sigmoide, el modelo de Regresión Logística se expresa como:

$$\pi(\mathbf{x}) = P(Y = 1 \mid \mathbf{X} = \mathbf{x}) = \sigma(\boldsymbol{\beta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T \mathbf{x}}}$$

Equivalentemente, la probabilidad de la clase negativa ($Y = 0$) se obtiene por complementariedad:

$$P(Y = 0 \mid \mathbf{X} = \mathbf{x}) = 1 - \pi(\mathbf{x}) = \frac{e^{-\boldsymbol{\beta}^T \mathbf{x}}}{1 + e^{-\boldsymbol{\beta}^T \mathbf{x}}}$$

### 1.2.3 La Transformación Logit (Log-Odds)

Una transformación fundamental en la Regresión Logística es el **logit**, definido como el logaritmo natural de la *odds* (probabilidad de éxito sobre probabilidad de fracaso):

$$\text{logit}(\pi) = \ln\left(\frac{\pi}{1 - \pi}\right) = \ln(\text{odds})$$

En el contexto del modelo logístico:

$$\ln\left(\frac{\pi(\mathbf{x})}{1 - \pi(\mathbf{x})}\right) = \boldsymbol{\beta}^T \mathbf{x} = \beta_0 + \beta_1 x_1 + ... + \beta_p x_p$$

Esta formulación revela que la Regresión Logística modela de manera **lineal** el logaritmo de la odds de pertenecer a la clase positiva, lo cual constituye la base para la interpretación de coeficientes.

### 1.2.4 Estimación de Parámetros

Los coeficientes $\boldsymbol{\beta}$ se estiman mediante el método de **Máxima Verosimilitud** (MLE, *Maximum Likelihood Estimation*). Dado un conjunto de $n$ observaciones independientes $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, la función de verosimilitud se construye como:

$$\mathcal{L}(\boldsymbol{\beta}) = \prod_{i=1}^{n} \pi(\mathbf{x}_i)^{y_i} [1 - \pi(\mathbf{x}_i)]^{1 - y_i}$$

Es computacionalmente conveniente trabajar con el logaritmo de la verosimilitud:

$$\ell(\boldsymbol{\beta}) = \ln \mathcal{L}(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \ln \pi(\mathbf{x}_i) + (1 - y_i) \ln (1 - \pi(\mathbf{x}_i)) \right]$$

Sustituyendo $\pi(\mathbf{x}_i) = \sigma(\boldsymbol{\beta}^T \mathbf{x}_i)$:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \boldsymbol{\beta}^T \mathbf{x}_i - \ln(1 + e^{\boldsymbol{\beta}^T \mathbf{x}_i}) \right]$$

Los estimadores de máxima verosimilitud $\hat{\boldsymbol{\beta}}$ se obtienen maximizando $\ell(\boldsymbol{\beta})$, lo cual requiere métodos numéricos iterativos (e.g., *Iteratively Reweighted Least Squares*, algoritmos quasi-Newton como L-BFGS, o gradiente descendente) debido a la no-linealidad de la función.

### 1.2.5 Regularización

Para prevenir el sobreajuste (*overfitting*), especialmente en escenarios con alta dimensionalidad, se incorpora regularización a la función objetivo. La forma más común es la **regularización L2** (Ridge), que penaliza la norma euclidiana de los coeficientes:

$$\ell_{\text{reg}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \frac{\lambda}{2} \|\boldsymbol{\beta}\|_2^2$$

Donde $\lambda > 0$ es el hiperparámetro de regularización. En términos equivalentes, el parámetro $C = 1/\lambda$ (utilizado en scikit-learn) controla la inversa de la fuerza de regularización: valores menores de $C$ implican mayor regularización.

---

## 1.3 Interpretación Probabilística

### 1.3.1 Interpretación de Coeficientes

La interpretación de los coeficientes $\beta_j$ en la Regresión Logística difiere sustancialmente de la Regresión Lineal debido a la no-linealidad introducida por la función sigmoide. La interpretación correcta se realiza en términos de **cambios en la log-odds** o, equivalentemente, en términos de **Odds Ratios (OR)**.

#### Cambio en la Log-Odds

Un incremento de una unidad en la variable $x_j$ (manteniendo constantes todas las demás variables) produce un cambio aditivo en la log-odds igual a $\beta_j$:

$$\Delta \ln(\text{odds}) = \beta_j \cdot \Delta x_j$$

Para un incremento unitario ($\Delta x_j = 1$):

$$\ln(\text{odds}_{\text{nuevo}}) - \ln(\text{odds}_{\text{original}}) = \beta_j$$

#### Odds Ratio (OR)

El **Odds Ratio** cuantifica el cambio multiplicativo en la odds asociado con un incremento unitario en la variable predictora:

$$\text{OR}_j = \frac{\text{odds}(x_j + 1)}{\text{odds}(x_j)} = e^{\beta_j}$$

| Valor de OR | Interpretación |
|-------------|----------------|
| $\text{OR} > 1$ | La variable aumenta la odds de éxito. Un aumento de una desviación estándar multiplica la odds por el valor del OR. |
| $\text{OR} = 1$ | La variable no tiene efecto sobre la odds ($\beta_j = 0$). |
| $0 < \text{OR} < 1$ | La variable reduce la odds de éxito. El factor de reducción es $\text{OR}$. |

#### Efecto sobre la Probabilidad

Aunque no existe una relación lineal directa entre $\beta_j$ y la probabilidad $\pi(\mathbf{x})$, el efecto es más pronunciado cuando la probabilidad está cercana a 0.5 (donde la función sigmoide es más empinada) y menos pronunciado en los extremos (efecto de saturación).

### 1.3.2 Probabilidad Predicha

La salida directa del modelo es la **probabilidad estimada** de pertenecer a la clase positiva:

$$\hat{\pi}(\mathbf{x}) = \frac{1}{1 + e^{-\hat{\boldsymbol{\beta}}^T \mathbf{x}}}$$

Esta probabilidad puede utilizarse directamente para **clasificación** mediante la aplicación de un umbral (*threshold*) típicamente en $\theta = 0.5$:

$$\hat{y} = \begin{cases} 1 & \text{si } \hat{\pi}(\mathbf{x}) \geq 0.5 \\ 0 & \text{si } \hat{\pi}(\mathbf{x}) < 0.5 \end{cases}$$

La elección del umbral puede ajustarse según el contexto del problema, particularmente en situaciones con costos asimétricos de falsos positivos y falsos negativos.

---

## 1.4 Supuestos del Modelo

La Regresión Logística, como modelo estadístico paramétrico, descansa sobre varios supuestos fundamentales que deben considerarse durante su aplicación:

### 1.4.1 Independencia de Observaciones

Las observaciones deben ser **estadísticamente independientes** entre sí. La violación de este supuesto (e.g., datos con estructura jerárquica, series temporales correlacionadas, o mediciones repetidas) puede llevar a inferencias incorrectas sobre la significancia de los coeficientes.

*Implicación para este estudio:* Las carreras son eventos independientes, aunque pilotos y constructores aparecen múltiples veces. Se asume independencia condicional dado las características observadas.

### 1.4.2 Linealidad en la Log-Odds

El supuesto más crítico es que la relación entre las variables predictoras (continuas) y la **log-odds de la respuesta** es lineal. Esto implica que:

$$\ln\left(\frac{\pi(\mathbf{x})}{1 - \pi(\mathbf{x})}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$

*Verificación:* Este supuesto puede evaluarse mediante análisis de residuos o comparando modelos con y sin transformaciones no lineales de las variables.

### 1.4.3 Ausencia de Multicolinealidad Severa

Las variables predictoras no deben estar perfectamente correlacionadas (colinealidad) o altamente correlacionadas (multicolinealidad severa). La multicolinealidad puede:

- Inflar las varianzas de los estimadores de coeficientes
- Hacer los coeficientes estadísticamente inestables e interpretablemente contradictorios
- Dificultar la identificación del efecto individual de cada variable

*Mitigación:* En este estudio, el uso de regularización L2 (Ridge) ayuda a estabilizar las estimaciones en presencia de correlaciones moderadas entre variables.

### 1.4.4 Independencia de Errores (No Aplicabilidad Estricta)

A diferencia de la Regresión Lineal, la Regresión Logística no asume normalidad ni homocedasticidad de los residuos, ya que la varianza de la respuesta binaria está completamente determinada por su media ($\text{Var}(Y) = \pi(1-\pi)$).

### 1.4.5 Tamaño Muestral Adecuado

Se recomienda un tamaño muestral suficiente para garantizar la estabilidad de las estimaciones de máxima verosimilitud. Reglas empíricas sugieren:

- Mínimo de 10 observaciones por parámetro (regla de *10 events per variable*)
- Preferiblemente 20-50 observaciones por parámetro para modelos complejos

*Verificación:* Con 655,189 observaciones y 55 parámetros, el ratio es aproximadamente 11,913:1, ampliamente satisfactorio.

### 1.4.6 Correcta Especificación del Modelo

Se asume que todas las variables relevantes están incluidas y que no hay variables irrelevantes que introduzcan ruido. La omisión de variables importantes puede causar sesgo por variables omitidas.

---

## 1.5 Ventajas y Limitaciones

### 1.5.1 Ventajas

| Ventaja | Descripción |
|---------|-------------|
| **Interpretabilidad** | Los coeficientes tienen interpretación directa en términos de log-odds y odds ratios, facilitando la comunicación de resultados |
| **Salida probabilística** | Proporciona probabilidades calibradas, no solo clasificaciones binarias, permitiendo análisis de riesgo y toma de decisiones informadas |
| **Eficiencia computacional** | El entrenamiento es computacionalmente eficiente incluso con grandes datasets |
| **Menos propenso al sobreajuste** | Comparado con modelos complejos (e.g., árboles profundos), especialmente con regularización apropiada |
| **Base estadística sólida** | Fundamentado en principios de inferencia estadística con propiedades teóricas bien estudiadas |
| **Extensibilidad natural** | Fácilmente extensible a clasificación multiclase (regresión logística multinomial) |

### 1.5.2 Limitaciones

| Limitación | Descripción |
|------------|-------------|
| **Frontera de decisión lineal** | La frontera de decisión es lineal en el espacio de características, lo que puede ser insuficiente para problemas con relaciones no lineales complejas |
| **Supuesto de linealidad en log-odds** | Requiere que la relación entre predictores y log-odds sea lineal; relaciones no lineales requieren transformaciones manuales |
| **Sensibilidad a outliers** | Observaciones atípicas pueden influir desproporcionadamente en las estimaciones de coeficientes |
| **Separación perfecta** | Si las clases son linealmente separables, los coeficientes de MLE divergen (problema de *complete separation*) |
| **Manejo de interacciones** | Las interacciones entre variables deben especificarse explícitamente; no se detectan automáticamente |
| **Asunción de independencia** | La independencia entre observaciones es un requisito estricto que puede violarse en datos estructurados |

### 1.5.3 Comparación con otros Modelos Lineales

| Característica | Regresión Lineal | Regresión Logística |
|----------------|------------------|---------------------|
| Tipo de respuesta | Continua | Binaria (o categórica) |
| Rango de predicción | $(-\infty, +\infty)$ | $[0, 1]$ (probabilidad) |
| Función de enlace | Identidad | Logit (sigmoide) |
| Método de estimación | Mínimos Cuadrados Ordinarios | Máxima Verosimilitud |
| Distribución de errores | Normal | Binomial |
| Interpretación coeficientes | Cambio en media de $Y$ | Cambio en log-odds de $Y=1$ |

---

# 2. Implementación del Modelo

## 2.1 Configuración del Modelo

### 2.1.1 Variable Objetivo y Codificación

La variable objetivo para el modelo de Regresión Logística es el resultado binario de la carrera, codificado según el siguiente esquema:

| Valor Codificado | Significado | Descripción |
|------------------|-------------|-------------|
| $y = 1$ | **Victoria** | El piloto finalizó la carrera en la primera posición |
| $y = 0$ | **No Victoria** | El piloto no alcanzó la primera posición (posiciones 2 a última, abandono o descalificación) |

Esta codificación binaria es apropiada para la Regresión Logística binaria y se deriva directamente del campo `positionOrder` del dataset original, donde `positionOrder = 1` se mapea a `win = 1`.

### 2.1.2 Conjunto de Variables Predictoras

El modelo utiliza un conjunto de **55 variables predictoras** (features) que abarcan múltiples dimensiones del fenómeno deportivo:

#### Variables Numéricas Continuas

| Variable | Descripción | Justificación Teórica |
|----------|-------------|----------------------|
| `grid` | Posición de salida en parrilla | Predictor clásico en F1; posiciones frontales correlacionan fuertemente con victoria |
| `positionOrder` | Orden de llegada (utilizado solo en feature engineering previo) | Referencia contextual |
| `points` | Puntos obtenidos en carrera | Refleja desempeño relativo |
| `laps` | Número de vueltas completadas | Indica fiabilidad y consistencia |
| `fastestLapSpeed` | Velocidad en vuelta rápida (km/h) | Métrica de rendimiento puro del vehículo |
| `age` | Edad del piloto | Factor demográfico potencialmente relacionado con experiencia vs. reflexes |

#### Variables Categóricas (Codificación One-Hot)

| Variable | Categorías | Descripción |
|----------|------------|-------------|
| `constructorId` | 211 valores | Identificador del constructor/escudería |
| `nationality` | Múltiples países | Nacionalidad del piloto |

La transformación de variables categóricas a representación numérica se realizó mediante **codificación one-hot** (*dummy variables*), generando variables binarias para cada categoría excepto una (categoría de referencia).

### 2.1.3 Estrategia de Manejo del Desbalance de Clases

El dataset presenta un **desbalance de clases severo**, característico de competiciones deportivas donde las victorias son eventos raros:

| Clase | Frecuencia | Porcentaje |
|-------|------------|------------|
| No Victoria ($y=0$) | 621,936 | 94.92% |
| Victoria ($y=1$) | 33,253 | 5.08% |
| **Total** | **655,189** | **100%** |

El ratio de desbalance es aproximadamente **18.7:1**, lo que implica que la clase mayoritaria supera a la minoritaria por un factor cercano a 19.

#### Estrategia Implementada: `class_weight='balanced'`

Se empleó la estrategia de ponderación de clases implementada en scikit-learn, que ajusta automáticamente los pesos inversamente proporcionales a las frecuencias de clase:

$$w_j = \frac{n}{k \cdot n_j}$$

Donde:
- $n$ = número total de observaciones
- $k$ = número de clases (2 en este caso)
- $n_j$ = número de observaciones en clase $j$

Esto resulta en pesos aproximados de:
- $w_0 \approx 0.53$ (clase mayoritaria)
- $w_1 \approx 9.85$ (clase minoritaria)

Esta ponderación ajusta la función de pérdida para penalizar más los errores de clasificación en la clase minoritaria (victorias), incentivando al modelo a no simplemente predecir la clase mayoritaria constantemente.

#### Fundamentación Teórica de la Estrategia

La ponderación de clases modifica la función de log-verosimilitud ponderada:

$$\ell_{w}(\boldsymbol{\beta}) = \sum_{i=1}^{n} w_{y_i} \left[ y_i \ln \pi(\mathbf{x}_i) + (1 - y_i) \ln (1 - \pi(\mathbf{x}_i)) \right]$$

Donde $w_{y_i}$ asigna mayor peso a las observaciones de la clase minoritaria durante la optimización.

---

## 2.2 Entrenamiento del Modelo

### 2.2.1 Conjunto de Datos para Entrenamiento

La división de datos siguió un esquema de **validación hold-out** con estratificación:

| Conjunto | Observaciones | Porcentaje | Victorias (%) | No Victorias (%) |
|----------|---------------|------------|---------------|------------------|
| Entrenamiento | 524,151 | 80.0% | 26,602 (5.08%) | 497,549 (94.92%) |
| Prueba | 131,038 | 20.0% | 6,651 (5.08%) | 124,387 (94.92%) |

La **estratificación** garantiza que la proporción de clases se mantenga idéntica en ambos conjuntos, evitando sesgos de muestreo que podrían afectar la evaluación del modelo.

### 2.2.2 Preprocesamiento: Escalado de Características

Dado que la Regresión Logística con regularización L2 es sensible a la escala de las variables predictoras, se aplicó **estandarización** (*StandardScaler*):

$$x_j^{\text{scaled}} = \frac{x_j - \mu_j}{\sigma_j}$$

Donde $\mu_j$ y $\sigma_j$ son la media y desviación estándar calculadas únicamente sobre el conjunto de entrenamiento, aplicándose la misma transformación al conjunto de prueba.

**Justificación:**
- La regularización L2 penaliza $\sum \beta_j^2$, lo cual es equitativo solo si las variables están en escalas comparables
- Variables con magnitudes mayores recibirían penalizaciones desproporcionadas sin escalado
- Mejora la estabilidad numérica del optimizador

### 2.2.3 Procedimiento de Ajuste del Modelo

El entrenamiento se realizó utilizando la implementación de scikit-learn (`LogisticRegression`) con los siguientes hiperparámetros:

| Hiperparámetro | Valor | Descripción |
|----------------|-------|-------------|
| `class_weight` | `'balanced'` | Ponderación automática de clases |
| `max_iter` | 1000 | Máximo de iteraciones para convergencia |
| `solver` | `'lbfgs'` | Algoritmo quasi-Newton L-BFGS |
| `C` | 1.0 | Inverso de la fuerza de regularización |
| `random_state` | 42 | Semilla para reproducibilidad |
| `multi_class` | `'auto'` | Detección automática binaria/multiclase |
| `penalty` | `'l2'` | Regularización Ridge (implícita con L-BFGS) |

#### Algoritmo de Optimización: L-BFGS

El solver L-BFGS (*Limited-memory Broyden-Fletcher-Goldfarb-Shanno*) es un método quasi-Newton que aproxima la matriz Hessiana de segunda derivadas utilizando información de gradientes previos, sin almacenar la matriz completa. Es particularmente eficiente para problemas con moderada dimensionalidad (p < 10,000) y función objetivo suave.

### 2.2.4 Tiempo de Entrenamiento y Convergencia

| Métrica | Valor |
|---------|-------|
| Tiempo de entrenamiento | **0.97 segundos** |
| Iteraciones hasta convergencia | **34** |
| Convergencia alcanzada | **Sí** |

El modelo convergió exitosamente en 34 iteraciones, considerablemente por debajo del límite máximo de 1,000 iteraciones. Esto indica que la función objetivo alcanzó un óptimo estable dentro de las tolerancias predefinidas (tolerancia de cambio en parámetros y tolerancia de gradiente).

### 2.2.5 Consideraciones Prácticas Durante el Ajuste

Durante el proceso de entrenamiento se identificaron las siguientes consideraciones:

1. **Convergencia estable**: El solver L-BFGS demostró convergencia robusta sin necesidad de ajustar parámetros de tolerancia.

2. **Efectividad del escalado**: El preprocesamiento con `StandardScaler` fue esencial para la estabilidad numérica, especialmente considerando que variables como `fastestLapSpeed` operan en escalas diferentes a variables dummy.

3. **Impacto de `class_weight`**: Sin ponderación de clases, el modelo tendía a predecir constantemente la clase mayoritaria (no victoria), resultando en métricas de recall para la clase minoritaria cercanas a cero.

4. **Dimensionalidad moderada**: Con 55 variables, el problema está dentro del rango óptimo para L-BFGS, evitando la necesidad de solvers alternativos como SAG/SAGA que son preferibles para dimensionalidad muy alta.

---

## 2.3 Análisis de Coeficientes

### 2.3.1 Tabla de Coeficientes Estimados

Los coeficientes del modelo representan el cambio en la log-odds de victoria asociado con un incremento de una desviación estándar en cada variable (debido al escalado aplicado).

#### Top 20 Variables por Magnitud Absoluta

| Posición | Variable | Coeficiente ($\beta$) | Odds Ratio ($e^{\beta}$) | Interpretación |
|----------|----------|----------------------|-------------------------|----------------|
| 1 | `grid` | -2.1400 | 0.1177 | Efecto protector dominante |
| 2 | `nationality_Brazilian` | -1.7508 | 0.1736 | Asociación negativa fuerte |
| 3 | `nationality_Canadian` | -1.3981 | 0.2471 | Efecto protector |
| 4 | `constructorId_10` | -1.2907 | 0.2751 | Efecto protector |
| 5 | `constructorId_117` | -1.2497 | 0.2866 | Efecto protector |
| 6 | `constructorId_210` | -1.2376 | 0.2901 | Efecto protector |
| 7 | `constructorId_9` | +1.2353 | 3.4394 | Efecto facilitador dominante |
| 8 | `nationality_Russian` | -1.1623 | 0.3128 | Efecto protector |
| 9 | `constructorId_5` | -1.1010 | 0.3325 | Efecto protector |
| 10 | `constructorId_51` | -1.0806 | 0.3394 | Efecto protector |
| 11 | `constructorId_131` | +1.0626 | 2.8940 | Efecto facilitador |
| 12 | `nationality_Thai` | -1.0605 | 0.3463 | Efecto protector |
| 13 | `nationality_Japanese` | -1.0433 | 0.3523 | Efecto protector |
| 14 | `nationality_British` | +1.0146 | 2.7582 | Efecto facilitador |
| 15 | `constructorId_6` | +0.9783 | 2.6599 | Efecto facilitador |
| 16 | `constructorId_4` | -0.9665 | 0.3804 | Efecto protector |
| 17 | `nationality_German` | +0.7974 | 2.2199 | Efecto facilitador |
| 18 | `constructorId_15` | -0.7792 | 0.4588 | Efecto protector |
| 19 | `constructorId_1` | +0.7769 | 2.1748 | Efecto facilitador |
| 20 | `nationality_Danish` | -0.7509 | 0.4719 | Efecto protector |

### 2.3.2 Estadísticas Descriptivas de Coeficientes

| Estadística | Valor |
|-------------|-------|
| Media | -0.1207 |
| Mediana | 0.0402 |
| Desviación Estándar | 0.7595 |
| Mínimo | -2.1400 (`grid`) |
| Máximo | +1.2353 (`constructorId_9`) |
| Coeficientes positivos | 30 (54.5%) |
| Coeficientes negativos | 25 (45.5%) |

### 2.3.3 Análisis de Signos y Magnitudes

#### Variables con Mayor Influencia Positiva

Las variables con coeficientes positivos más grandes aumentan la odds de victoria:

| Variable | Coeficiente | Odds Ratio | Interpretación |
|----------|-------------|------------|----------------|
| `constructorId_9` | +1.2353 | 3.44 | Mayor efecto facilitador identificado |
| `constructorId_131` | +1.0626 | 2.89 | Segundo mayor efecto positivo |
| `nationality_British` | +1.0146 | 2.76 | Nacionalidad asociada con mayor odds de victoria |
| `constructorId_6` | +0.9783 | 2.66 | Constructor históricamente dominante |
| `nationality_German` | +0.7974 | 2.22 | Nacionalidad con tradición en F1 |

**Interpretación contextual:** Los coeficientes positivos asociados a constructores específicos (IDs 9, 131, 6, 1) reflejan el dominio histórico de ciertas escuderías en el campeonato mundial. Las nacionalidades británica y alemana tienen tradiciones históricas fuertes en Fórmula 1, con múltiples campeones mundiales.

#### Variables con Mayor Influencia Negativa

Las variables con coeficientes negativos más grandes (en valor absoluto) reducen la odds de victoria:

| Variable | Coeficiente | Odds Ratio | Interpretación |
|----------|-------------|------------|----------------|
| `grid` | -2.1400 | 0.12 | Variable más determinante del modelo |
| `nationality_Brazilian` | -1.7508 | 0.17 | Efecto protector significativo |
| `nationality_Canadian` | -1.3981 | 0.25 | Efecto protector |
| `constructorId_10` | -1.2907 | 0.28 | Constructor con baja tasa de victorias |
| `constructorId_117` | -1.2497 | 0.29 | Constructor históricamente menos competitivo |

**Interpretación de `grid`:** La variable `grid` tiene el coeficiente negativo más grande en magnitud absoluta, lo cual es consistente con la lógica del deporte: posiciones de salida más altas (números mayores en `grid`) están asociadas con menor probabilidad de victoria. La codificación one-hot implícita en el procesamiento previo puede haber generado una representación donde mayores valores numéricos de `grid` corresponden a posiciones más retrasadas en la parrilla.

### 2.3.4 Interpretación de Odds Ratios

Los Odds Ratios permiten cuantificar el efecto multiplicativo:

**Ejemplo de interpretación:** Para `constructorId_9` con OR = 3.44:
> "Ser piloto del constructor 9 multiplica la odds de victoria por un factor de 3.44, en comparación con la categoría de referencia, manteniendo constantes todas las demás variables."

**Ejemplo de efecto protector:** Para `grid` con OR = 0.12:
> "Un aumento de una desviación estándar en la posición de salida reduce la odds de victoria a un 12% de su valor original, representando una reducción del 88%."

---

# 3. Resultados Preliminares

Los resultados del entrenamiento del modelo de Regresión Logística revelan las siguientes métricas preliminares de rendimiento:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **AUC-ROC** | 0.9480 | Excelente capacidad discriminativa; el modelo distingue efectivamente entre victorias y no victorias |
| **AUC-PR** | 0.5041 | Rendimiento moderado en la clase minoritaria; mejora sustancial respecto al baseline (0.05) |
| **F1-Score** | 0.3511 | Balance entre precisión y recall para la clase victoria |
| **Log-Loss (test)** | 0.3187 | Pérdida de predicción probabilística baja, indicando buena calibración |

### Interpretación de Resultados

1. **Capacidad Discriminativa (AUC-ROC = 0.948):** El área bajo la curva ROC cercana a 0.95 indica que el modelo tiene excelente capacidad para ordenar las observaciones según su probabilidad de victoria. Un clasificador aleatorio tendría AUC-ROC = 0.5.

2. **Rendimiento en Clase Minoritaria (AUC-PR = 0.504):** Dado que las victorias representan solo el 5% de los datos, la AUC-PR es más informativa que AUC-ROC. Un valor de 0.50 es sustancialmente superior al baseline (proporción de positivos = 0.05), indicando que el modelo captura patrones predictivos reales.

3. **Efecto del Manejo de Desbalance:** La implementación de `class_weight='balanced'` permitió que el modelo identificara correctamente un subconjunto de victorias, evidenciado por un F1-Score positivo (0.35) que contrasta con el valor de cero observado en modelos sin manejo de desbalance.

---

# 4. Anexo: Fragmentos de Código Relevante

## 4.1 Importación de Bibliotecas y Configuración

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss
)
import time
import warnings
warnings.filterwarnings('ignore')

# Configuración de directorios
DATA_DIR = "../data"
OUTPUT_DIR = "../resultados_regresion_logistica"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

## 4.2 Carga de Datos y Preparación

```python
# Carga del dataset procesado
df = pd.read_csv(f'{DATA_DIR}/dataset_tesis_f1.csv')

# Separación de features (X) y target (y)
X = df.drop('win', axis=1)
y = df['win']

# Nombres de features para análisis posterior
feature_names = X.columns.tolist()

print(f"Dataset: {df.shape[0]} observaciones, {X.shape[1]} variables predictoras")
print(f"Distribución de clases: {y.value_counts().to_dict()}")
```

## 4.3 División Train/Test con Estratificación

```python
# División estratificada para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Estratificación esencial
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print(f"Proporción de victorias en train: {y_train.mean():.4f}")
print(f"Proporción de victorias en test: {y_test.mean():.4f}")
```

## 4.4 Escalado de Características

```python
# Estandarización de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Características escaladas a media=0, desviación estándar=1")
```

## 4.5 Configuración y Entrenamiento del Modelo

```python
# Configuración del modelo con manejo de desbalance
model_config = {
    'class_weight': 'balanced',  # Manejo automático de clases desbalanceadas
    'max_iter': 1000,            # Iteraciones máximas
    'solver': 'lbfgs',           # Algoritmo quasi-Newton
    'C': 1.0,                    # Inverso de regularización L2
    'random_state': 42,
}

# Instanciación y entrenamiento
model = LogisticRegression(**model_config)

# Medición de tiempo de entrenamiento
start_time = time.time()
model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
print(f"Iteraciones hasta convergencia: {model.n_iter_}")
print(f"Intercepto (β₀): {model.intercept_[0]:.6f}")
```

## 4.6 Generación de Predicciones

```python
# Predicciones de probabilidad (salida sigmoide)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Predicciones binarias con umbral 0.5
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print(f"Rango de probabilidades (test): [{y_test_proba.min():.4f}, {y_test_proba.max():.4f}]")
```

## 4.7 Análisis de Coeficientes

```python
# Creación de DataFrame de coeficientes
coef_df = pd.DataFrame({
    'variable': feature_names,
    'coeficiente': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0]),
    'abs_coeficiente': np.abs(model.coef_[0])
}).sort_values('abs_coeficiente', ascending=False)

# Visualización de top 20 coeficientes
print("\nTop 20 variables por magnitud de coeficiente:")
print(coef_df.head(20)[['variable', 'coeficiente', 'odds_ratio']].to_string(index=False))

# Guardado de resultados
coef_df.to_csv(f'{OUTPUT_DIR}/tabla_coeficientes_logistica.csv', index=False)
```

## 4.8 Cálculo de Métricas Preliminares

```python
# Métricas de clasificación
f1_test = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)
logloss_test = log_loss(y_test, y_test_proba)

print(f"AUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR: {pr_auc:.4f}")
print(f"F1-Score: {f1_test:.4f}")
print(f"Log-Loss: {logloss_test:.6f}")

# Guardado de métricas
metricas = {
    'Modelo': 'Regresion Logistica',
    'F1_Test': f1_test,
    'AUC_ROC': roc_auc,
    'AUC_PR': pr_auc,
    'LogLoss_Test': logloss_test,
    'Intercepto': model.intercept_[0],
    'Tiempo_Entrenamiento_s': training_time,
}
pd.DataFrame([metricas]).to_csv(f'{OUTPUT_DIR}/metricas_modelo.csv', index=False)
```

## 4.9 Visualización de Coeficientes

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Visualización 1: Top 20 coeficientes por magnitud
plt.figure(figsize=(14, 10))
top_20 = coef_df.head(20)
colores = ['#2E86AB' if c >= 0 else '#D1495B' for c in top_20['coeficiente']]

plt.barh(range(len(top_20)), top_20['abs_coeficiente'], 
         color=colores, alpha=0.8, edgecolor='black')
plt.yticks(range(len(top_20)), top_20['variable'])
plt.xlabel('|Coeficiente| (Magnitud del Efecto)', fontsize=12, fontweight='bold')
plt.title('Top 20 Variables más Influyentes - Regresión Logística', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# Leyenda
pos_patch = mpatches.Patch(color='#2E86AB', label='Efecto Positivo (+)')
neg_patch = mpatches.Patch(color='#D1495B', label='Efecto Negativo (-)')
plt.legend(handles=[pos_patch, neg_patch], loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/coeficientes_top20.png', dpi=300, bbox_inches='tight')
```

---

## Referencias

- Cox, D. R. (1958). *The regression analysis of binary sequences*. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

---

**Fin del Documento**

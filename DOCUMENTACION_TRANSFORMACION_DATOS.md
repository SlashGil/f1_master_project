# Documentación Metodológica y Técnica: Preparación del Dataset

**Versión:** 3.0 (Post-Auditoría Técnica)  
**Fecha:** Septiembre 2026  
**Autor:** Salvador Romero Gil  
**Estado:** Auditado y Validado (Escenario Pre-Carrera Sin Fuga Temporal)

---

## Resumen Ejecutivo

Este documento detalla el proceso metodológico y técnico para la preparación del conjunto de datos unificado utilizado en los modelos predictivos de victorias en la Fórmula 1. 

A raíz de la **Auditoría Técnica y Diagnóstico del Pipeline de Datos (Septiembre 2026)**, el pipeline ha sido completamente estandarizado para garantizar la validez científica y evitar toda forma de **fuga de información (data leakage)**:

1. **Escenario Pre-Carrera Estricto**: Se eliminan todas las variables que se conocen durante o después de la carrera (`fastestLapSpeed`, `milliseconds_pit_stop`, `milliseconds_lap_time`, `positionOrder`).
2. **Exclusión de `raceId` de los Predictores**: `raceId` actúa como identificador relacional y proxy temporal para la ordenación cronológica, pero queda **estrictamente excluido** de la matriz de características $X$.
3. **Partición Cronológica Temporal (80/20)**: Se sustituyó definitivamente la partición aleatoria `train_test_split(shuffle=True)` por un corte temporal donde el 80% pasado entrena el modelo y el 20% futuro se evalúa como prueba.
4. **Preprocesamiento y Escalado sin Fuga**: El escalado (`StandardScaler`) se calibra (`fit`) **únicamente** sobre el conjunto de entrenamiento (`X_train`) y se aplica (`transform`) sobre el conjunto de prueba (`X_test`).
5. **Fuente Única de Verdad**: El archivo [`data/dataset_tesis_f1.csv`](data/dataset_tesis_f1.csv) consolida exactamente **25,121 observaciones** para todos los modelos del proyecto.

---

## 1. Obtención y Descripción de las Fuentes de Datos

### 1.1. Fuente Original
Los datos primarios provienen de la **Ergast Developer API** (base histórica oficial de la Fórmula 1 desde 1950 hasta la actualidad).

### 1.2. Archivos Empleados en el Pipeline Pre-Carrera

Para construir la base analítica pre-carrera se utilizan las tablas base del dominio:

| Archivo | Observaciones Aprox. | Rol en el Pipeline |
|---|---|---|
| `results.csv` | 26,759 | **Tabla Central:** Resultados por piloto y carrera. Fuente de `grid` y `win` (`positionOrder == 1`). |
| `drivers.csv` | 861 | Datos demográficos del piloto (`dob`, `nationality`). |
| `races.csv` | 1,125 | Catálogo de carreras (`year`, `round`, `date`, `circuitId`). |
| `constructors.csv` | 212 | Catálogo de escuderías (`constructorId`, `name`). |
| `circuits.csv` | 77 | Catálogo de circuitos (`circuitId`). |

*(Nota: Tablas como `lap_times.csv` y `pit_stops.csv` fueron excluidas del modelado predictivo base por contener únicamente información posterior al inicio de la carrera).*

---

## 2. Integración y Limpieza de Datos

### 2.1. Fusión de Tablas (Joins)
La fusión se realiza preservando la integridad referencial completa:
```python
data = results.merge(
    drivers[['driverId', 'nationality', 'dob']],
    on='driverId',
    how='inner'
).merge(
    races[['raceId', 'year', 'round', 'date', 'circuitId']],
    on='raceId',
    how='inner'
)
```

### 2.2. Normalización de Nulos y Filtros de Calidad
1. **Reemplazo de `\N`**: Se reemplaza la cadena `\N` de PostgreSQL por `numpy.nan`.
2. **Filtro de Parrilla Válida (`grid > 0`)**: Se descartan salidas desde el pit lane sin posición fija o descalificaciones previas.
3. **Cálculo de Edad al Momento de la Carrera**:
   $$\text{age} = \frac{\text{fecha\_carrera} - \text{fecha\_nacimiento}}{365.25}$$

---

## 3. Definición de la Variable Objetivo ($y$)

- **Definición**: Variable binaria de victoria (`win`):
  $$y = \begin{cases} 1 & \text{si } \text{positionOrder} = 1 \\ 0 & \text{en otro caso} \end{cases}$$
- **Distribución de Clases (25,121 observaciones)**:
  - **Victorias ($y=1$)**: 1,128 (4.49%)
  - **No Victorias ($y=0$)**: 23,993 (95.51%)
  - **Ratio de Desbalance**: 1:21.3 (Severo)

---

## 4. Matriz de Características ($X$) y Reglas Anti-Leakage

### 4.1. Variables Predictoras Autorizadas (Pre-Carrera)

| Variable | Tipo | Justificación Metodológica |
|---|---|---|
| `grid` | Numérica Discreta | Posición de salida en parrilla tras sesión de clasificación oficial. |
| `age` | Numérica Continua | Edad exacta del piloto el día del Gran Premio. |
| `year` | Numérica / Contextual | Temporada del campeonato. |
| `round` | Numérica Discreta | Ronda dentro de la temporada (avance del campeonato). |
| `nationality_*` | Dummy (One-Hot) | Nacionalidad del piloto. |
| `constructorId_*` | Dummy (One-Hot) | Identificador de la escudería / constructor. |

### 4.2. Variables Prohibidas y Excluidas
- `positionOrder`, `position`, `points`, `statusId` (Target leakage / post-carrera).
- `fastestLapSpeed`, `fastestLapTime`, `rank` (Post-carrera).
- `milliseconds_pit_stop`, `milliseconds_lap_time` (Intra-carrera).
- **`raceId`**: Reservado exclusivamente para la ordenación cronológica interna; **excluido de la matriz de predictores**.

---

## 5. Estrategia de Partición y Validación Temporal

### 5.1. Ordenación Cronológica y Partición 80/20
Los datos se ordenan estrictamente por tiempo:
```python
df_sorted = df.sort_values(['year', 'round', 'raceId']).reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

train_df = df_sorted.iloc[:split_idx]  # 20,096 observaciones (Pasado)
test_df  = df_sorted.iloc[split_idx:]  #  5,025 observaciones (Futuro)

# Exclusión explícita de raceId y win
X_train = train_df.drop(columns=['win', 'raceId'], errors='ignore')
y_train = train_df['win']
X_test  = test_df.drop(columns=['win', 'raceId'], errors='ignore')
y_test  = test_df['win']
```

### 5.2. Escalado de Características
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit ÚNICAMENTE en train
X_test_scaled  = scaler.transform(X_test)        # Transform en test
```

---

## 6. Modelos Implementados en el Proyecto

Todos los scripts bajo [`modelos/`](modelos/) consumen de manera homogénea el dataset unificado respetando las reglas de la auditoría:

1. **[`modelos/1_regresion_lineal.py`](modelos/1_regresion_lineal.py)**: Regresión lineal base para análisis de coeficientes y magnitud de factores pre-carrera (sin `raceId`).
2. **[`modelos/2_random_forest.py`](modelos/2_random_forest.py)**: Random Forest con `TimeSeriesSplit` en validación cruzada y `min_samples_leaf=2`.
3. **[`modelos/3_perceptron_multicapa.py`](modelos/3_perceptron_multicapa.py)**: Red neuronal PyTorch con corrección de `BCELoss(reduction='none')` para class weighting efectivo por muestra.
4. **[`modelos/4_regresion_logistica.py`](modelos/4_regresion_logistica.py)**: Regresión logística con Odds Ratios interpretables (sin distorsión de `raceId`).

---

## 7. Higiene del Repositorio y Scripts Legacy

Los scripts antiguos que utilizaban `train_test_split(shuffle=True)` y variables intra-carrera han sido trasladados a la carpeta [`legacy/`](legacy/) con su respectiva documentación histórica.

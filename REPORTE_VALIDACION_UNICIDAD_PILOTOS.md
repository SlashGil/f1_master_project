# Dictamen de Validación: Unicidad Piloto-Carrera en `dataset_tesis`
**Programa:** Maestría en Ciencia de Datos / Inteligencia Artificial  
**Proyecto:** Predicción Pre-Carrera de Victoria en Fórmula 1 mediante Machine Learning  
**Fecha de Emisión Oficial:** 07 de Octubre de 2026 (Auditoría de Integridad Muestral)  
**Estado de Validación:** `APROBADO [PASS]`

---

## 1. Declaración del Principio de Unicidad

En el diseño muestral para la predicción de resultados en Fórmula 1 previa al inicio de la competencia ($t_0$), cada fila de datos debe corresponder biyectivamente a una **unidad de observación atómica**:
$$\forall r \in \mathcal{R}, \forall d \in \mathcal{D}_r : N(r, d) = 1$$
Donde:
* $\mathcal{R}$ es el conjunto total de Grandes Premios oficiales analizados ($N_{races} = 1,125$).
* $\mathcal{D}_r$ es el conjunto de pilotos que tomaron formalmente la salida en la parrilla con posición válida ($grid > 0$).
* $N(r, d)$ representa el número de registros en la base de datos para el piloto $d$ en la carrera $r$.

---

## 2. Resultados de la Auditoría Numérica

| Métrica / Dimensión Evaluada | Valor Observado | Valor Esperado / Teórico | Estado |
| :--- | :---: | :---: | :---: |
| **Total de Grandes Premios Auditados** | **1,125** | 1,125 carreras (1950-2024) | `CONFORME` |
| **Total de Pilotos Únicos en la Historia** | **792** | Catálogo oficial Ergast | `CONFORME` |
| **Duplicados en Conjunto de Prueba (Test 2012-2024)** | **0** | 0 duplicados | `APROBADO [100.0%]` |
| **Duplicados en Era Moderna (1965-2024)** | **0** | 0 duplicados | `APROBADO [100.0%]` |
| **Pares Evaluados con Exactamente 1 Registro** | **25,033** | 25,033 | `APROBADO [100.0%]` |
| **Máximo de Registros por Piloto por Carrera** | **1** | 1 | `CONFORME` |
| **Mínimo de Registros por Piloto por Carrera** | **1** | 1 | `CONFORME` |
| **Carreras con Más de un Ganador ($win=1$)** | **0** | 0 | `CONFORME` |

---

## 3. Desglose Temporal y Contextualización Histórica

### 3.1. Conjunto de Evaluación Ciega (Test Set 2012-2024)
* **Carreras analizadas:** 249 Grandes Premios (desde el GP de EE.UU. 2012 en Austin hasta Abu Dabi 2024).
* **Observaciones:** 5,013 registros de pilotos participantes.
* **Tasa de duplicados:** **0.0000%** (Cero duplicados observados).
* **Conclusión:** La integridad muestral sobre la que se calculan las métricas oficiales de la tesis (AUC-ROC, AUC-PR, F1-Score) está **100% blindada contra duplicidad**.

### 3.2. Era Moderna de la Fórmula 1 (1965-2024)
* Abarca 994 Grandes Premios y 22,217 registros.
* La prohibición reglamentaria de la FIA sobre coches compartidos garantiza que no existe ningún registro duplicado nativo en 60 años ininterrumpidos de competencia.

### 3.3. Era Primitiva y Regulación de Relevos Compartidos (1950-1964)
* En las primeras décadas del campeonato, la FIA permitía la figura deportiva de *shared drives*. Si el monoplaza de un piloto sufría una falla mecánica durante la carrera, este podía tomar el monoplaza de un compañero de equipo.
* La base de datos Ergast registra estos relevos asignando al piloto tanto su coche original de largada como el coche al que se subió en boxes, lo que generó **82 instancias históricas** (88 filas excedentes en 41 carreras).
* **Criterio Metodológico Pre-Carrera:** Debido a que el modelo predice la probabilidad de triunfo en $t_0$ (antes de la largada), la reasignación de coches durante la carrera representa un evento posterior a la largada. Por consiguiente, retener la posición inicial de largada en parrilla (`grid > 0`) elimina la duplicidad y restituye la biyección matemática:
  $$N(r, d) = 1 \quad \forall r \in \mathcal{R}, \forall d \in \mathcal{D}_r$$

---

## 4. Conclusión Académica

Se dictamina que el dataset cumple con el requisito de integridad estructural: **por cada carrera existe un solo registro para cada piloto que pudiese haber corrido dicha carrera**.

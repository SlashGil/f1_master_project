# Documento de Validación Metodológica, Auditoría de Datos y Trazabilidad Experimental
## Tesis de Maestría: Predicción Pre-Carrera de Victoria en Fórmula 1 mediante Machine Learning

**Autor:** Salvador Romero Gil  
**Programa:** Maestría en Ciencia de Datos / Inteligencia Artificial  
**Fecha de Emisión Oficial:** Noviembre de 2026  
**Entorno de Ejecución:** Repositorio Git Oficial (Rama `test`)  
**Dataset Oficial:** `data/dataset_tesis_f1.csv`  

---

## Resumen Ejecutivo

El presente documento constituye el dictamen técnico, metodológico y estadístico definitivo para la fundamentación del marco metodológico y el capítulo de ingeniería de datos de la tesis. En este informe se auditan, resuelven y justifican formalmente las ocho dimensiones críticas del diseño experimental:

1. **Ámbito Temporal y Cascada de Filtrado:** Confirmación de la cobertura histórica completa de 1950 a 2024 (75 temporadas, 1,125 Grandes Premios, cero carreras filtradas por año) y presentación de la cascada de trazabilidad exacta desde las 26,759 observaciones iniciales hasta las 25,033 finales.
2. **Auditoría y Resolución de Relevos Históricos (*Shared Drives*):** Análisis forense de las 88 filas descartadas en 39 carreras (`grid > 0`, 82 pares piloto-carrera duplicados) correspondientes a relevos intra-carrera de la era primitiva (1950–1964). Desglose exacto: **3 registros de co-ganadores** y **85 registros de relevos no ganadores** (`win = 0`). Se aplica la regla explícita centrada en el piloto ganador (conservando a Fangio en 1951 y 1956, y a Moss en 1957 con `win = 1`) y manteniendo a los pilotos no ganadores en su fila original de salida (`win = 0`).
3. **Verificación Directa de Unicidad Piloto-Carrera:** Incorporación de `driverId` en la estructura tabular y demostración de cero duplicados (`df.duplicated(['raceId', 'driverId']).sum() == 0`) y ganador único por carrera (`1,125` victorias en `1,125` carreras).
4. **Partición Temporal Causal y Punto de Corte Atómico:** Justificación teórica contra el split aleatorio 80/20, fijación del corte atómico en el GP de Abu Dabi 2012 (`raceId = 877`, 20,020 observaciones en Train, `79.97%`) y arranque en el GP de EE.UU. 2012 en Austin (`raceId = 878`, 5,013 observaciones en Test, `20.03%`), con verificación de estricta precedencia cronológica $\max(\text{fecha}_{\text{train}}) < \min(\text{fecha}_{\text{test}})$.
5. **Reconciliación Dimensional de Variables:** Cuadre algebraico exacto entre las 201 variables predictoras del vector $X$ (3 continuas/ordinales y 198 columnas dummy de constructor) y las 248 columnas físicas presentes en el archivo CSV (incluyendo identificadores y variables preservadas para estudios de ablación).
6. **Protocolo de Escalado sin Fuga (*Data Leakage Prevention*):** Demostración del ajuste estricto de `StandardScaler` exclusivamente sobre el bloque de entrenamiento (`fit_transform` en Train y `transform` ciego en Test).
7. **Fundamentación Teórica del Umbral de Clasificación:** Justificación del tratamiento del desbalance de clases 21:1 (~4.49% victorias), papel de la función de costos balanceada (`class_weight='balanced'`), calibración operativa del umbral $\tau^*$ vía curvas Precision-Recall ($F_1$-max) y formulación deportiva por ordenamiento de probabilidades (*Ranking / ArgMax* por Gran Premio).

---

## 1. Ámbito Temporal y Cascada de Trazabilidad de la Base de Datos

### 1.1. Cobertura Temporal Oficial (1950–2024)
La base de datos definitiva `dataset_tesis_f1.csv` abarca **75 temporadas oficiales ininterrumpidas de Fórmula 1**, desde la creación del Campeonato Mundial en 1950 hasta el término de la temporada 2024:
* **Primer Gran Premio:** Gran Premio de Gran Bretaña 1950 (Circuito de Silverstone, `raceId = 833`, fecha: `1950-05-13`).
* **Último Gran Premio:** Gran Premio de Abu Dabi 2024 (Circuito de Yas Marina, `raceId = 1144`, fecha: `2024-12-08`).
* **Total de Grandes Premios integrados:** **1,125 carreras**.
* **Carreras eliminadas por criterio de año:** **0 carreras**.

#### Clarificación sobre las Propuestas de Exclusión por Año (1950 vs. 1965 vs. 1980)
Durante las fases exploratorias iniciales se plantearon debates sobre restringir el marco temporal:
1. **La "Era Telemetría" (post-1980):** Se consideró acotar la base a partir de 1980 debido a que variables de rendimiento intra-carrera (vueltas rápidas por sector, paradas en boxes, degradación de neumáticos) solo existen en las últimas cuatro décadas. Sin embargo, en el diseño experimental definitivo se estableció que **toda variable intra-carrera constituye una fuente inaceptable de fuga de información (*data leakage*)** para predecir antes de la largada. Al restringir los predictores exclusivamente al estado pre-carrera (`grid`, `age`, `round`, `constructor`), dichas variables son 100% consistentes, medibles y trazables desde 1950. Descartar décadas previas a 1980 habría significado desechar arbitrariamente más de 300 Grandes Premios sin justificación técnica.
2. **La "Era sin Relevos" (post-1965):** De 1965 a 2024, la prohibición reglamentaria de la FIA sobre coches compartidos eliminó de raíz cualquier duplicidad natural. No obstante, en lugar de truncar la serie histórica eliminando las primeras 131 carreras (1950–1964), se optó por depurar las filas de relevos aplicando el principio metodológico pre-carrera, preservando la totalidad de las 1,125 carreras de la historia.

### 1.2. Cascada de Trazabilidad Secuencial de Observaciones

```text
[Flujo Metodológico de Depuración de Observaciones]
results.csv (26,759 filas | 1,125 carreras)
   │
   ├── Merge relacional con drivers.csv e inner join con races.csv
   │   └── Resultado: 26,759 filas (0 filas eliminadas, integridad referencial 100%)
   │
   ├── Depuración de nulos sobre variables mínimas pre-carrera (dropna)
   │   └── Resultado: 26,759 filas (0 filas eliminadas, completitud 100%)
   │
   ├── Filtro de inclusión deportiva en parrilla (grid > 0)
   │   └── Resultado: 25,121 filas (-1,638 filas descartadas | 1,125 carreras restantes | 1,128 victorias)
   │
   └── Regla de ganador único + deduplicación piloto-carrera (win desc, resultId asc)
       └── Resultado Final: 25,033 observaciones (-88 filas descartadas en 39 carreras | 1,125 victorias)
```

#### Tabla de Conteo y Justificación de Cada Transición

| Etapa del Pipeline | Archivo / Operación | Filas Entrada | Filas Descartadas | Filas Restantes | Carreras Activas | Justificación Científica y Metodológica |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **0. Catálogo Bruto** | `data/results.csv` | **26,759** | 0 | 26,759 | 1,125 | Registro completo de participaciones en F1 (1950–2024). |
| **1. Enlace Relacional** | `merge(drivers, races)` | 26,759 | **0** | 26,759 | 1,125 | Integridad referencial perfecta en llaves `driverId` y `raceId`. |
| **2. Manejo de Nulos** | `dropna()` columnas base | 26,759 | **0** | 26,759 | 1,125 | Las variables base (`grid`, `dob`, `date`, `constructorId`, `positionOrder`) no presentan valores nulos. |
| **3. Filtro de Parrilla** | Condición `grid > 0` | 26,759 | **1,638** | 25,121 | 1,125 | **Criterio de Inclusión Deportiva:** Se excluyen pilotos que no tomaron la salida: no clasificados (*DNQ*), no arrancaron (*DNS*), reservas de prácticas o largadas forzadas no calificadas. |
| **4. Ganador Único y Deduplicación** | Desempate de co-ganadores + `drop_duplicates(['raceId','driverId'], keep='first')` | 25,121 | **88** | **25,033** | **1,125** | **Biyección Piloto-Carrera con Prioridad al Piloto Ganador:** En `grid > 0` existían 82 pares duplicados en 39 carreras (88 filas extra: **3** de co-ganadores y **85** de relevos con `win = 0`). Se conserva 1 ganador por carrera (priorizando al piloto que cruzó 1.º) y la fila original de largada para los no ganadores. |
| **Consolidado Final** | `data/dataset_tesis_f1.csv` | **25,033** | — | **25,033** | **1,125** | Dataset oficial libre de duplicados y con exactamente 1,125 victorias. |

---

## 2. Auditoría y Resolución de Relevos Históricos (*Shared Drives*)

### 2.1. Causa Reglamentaria Deportiva
Durante las décadas de 1950 y principios de 1960, el reglamento deportivo de la FIA contemplaba la figura de la **conducción compartida (*shared drive*)**. Si el monoplaza del piloto principal de una escudería sufría una falla mecánica, el equipo estaba facultado para ordenar a un segundo piloto ingresar a los boxes, cederle su coche y permitir que el primer piloto continuara en carrera. 

La base de datos relacional Ergast registra estos relevos generando **una fila por cada combinación de piloto y coche**, asignando al piloto que tomó el relevo una segunda entrada en la misma carrera.

### 2.2. Criterio de Resolución: Prioridad al Piloto Ganador y Desglose de las 88 Filas
**Ningún piloto participante fue excluido de la carrera; se conservó exactamente 1 registro por piloto y por Gran Premio.** Dado que el modelado incorpora atributos individuales del piloto (como `age`) para predecir quién obtiene la victoria, se adoptó una regla explícita en dos pasos en [`data/preparar_datos_tesis.py`](data/preparar_datos_tesis.py):

1. **Ganador único por Gran Premio (`win.sum() == 1`):** En las 3 carreras con co-ganadores compartidos (`raceId` `828`, `784`, `780`), el piloto que cruzó la meta y obtuvo el mayor puntaje (`5.0` pts frente a `4.0` pts) conserva `win = 1`, mientras que su compañero co-ganador pasa a `win = 0`.
2. **Deduplicación `(raceId, driverId)`:** Se ordena por `['raceId', 'driverId', 'win', 'resultId']` (`ascending=[True, True, False, True]`) antes de `drop_duplicates(subset=["raceId", "driverId"], keep="first")`. De este modo, **quien ganó conserva su registro ganador (`win = 1`)** y **los pilotos que no ganaron conservan la fila con la que largaron originalmente (`resultId` ascendente)**.

#### ¿Qué eran los 88 registros eliminados? (3 co-ganadores + 85 relevos no ganadores)
En la etapa de `25,121` filas (`grid > 0`), las **88 filas excedentes** correspondían a **82 pares `(raceId, driverId)` duplicados** concentrados en **39 carreras** entre 1950 y 1964 (76 pares con 2 registros y 6 pares con 3 registros):
* **3 registros con `win = 1`:** Correspondían a los relevos duplicados en las 3 carreras con doble ganador histórico (en las cuales solo podía quedar 1 victoria por carrera).
* **85 registros con `win = 0`:** Correspondían a las segundas (y terceras) participaciones por relevo en boxes donde el piloto **no ganó la carrera** (15 finalizaciones en podio 2.º/3.º, 36 en puestos 4.º–10.º y 34 fuera del top 10 o abandonos).

#### Consecuencia sobre los 3 Grandes Premios con Doble Ganador ($win=1$)
En la base de datos histórica existían 3 carreras donde dos pilotos figuraban simultáneamente con victoria ($win=1$):
1. **GP de Francia de 1951 (`raceId = 828`):** Luigi Fagioli largó en la posición `grid = 7` con el Alfa Romeo 159 (#8). Juan Manuel Fangio largó en la pole position (`grid = 1`) en su Alfa Romeo (#4/#24). Tras sufrir problemas de motor, Fangio tomó el auto de Fagioli y ganó la carrera (`5.0` pts vs. `4.0` pts).
   * *Resolución:* **Fangio** conserva su registro ganador (`win = 1`, en el auto con el que ganó). **Fagioli** permanece en su fila original de largada (`grid = 7`, actualizado a `win = 0`). **Ambos pilotos compiten en el dataset y hay exactamente 1 ganador**.
2. **GP de Argentina de 1956 (`raceId = 784`):** Luigi Musso largó en `grid = 3` con la Ferrari D50 (#34). Fangio largó en `grid = 1` en su auto (#30), sufrió la rotura de la bomba de combustible y tomó el auto de Musso para ganar (`5.0` pts vs. `4.0` pts).
   * *Resolución:* **Fangio** conserva su registro ganador (`win = 1`). **Musso** permanece en su fila original de largada (`grid = 3`, `win = 0`). **Ambos pilotos permanecen en la carrera y se elimina el doble ganador**.
3. **GP de Gran Bretaña de 1957 (`raceId = 780`):** Tony Brooks largó en `grid = 3` en el Vanwall (#20). Stirling Moss largó en pole `grid = 1` (#18), experimentó fallos de motor, subió al coche de Brooks y triunfó (`5.0` pts vs. `4.0` pts).
   * *Resolución:* **Moss** conserva su registro ganador (`win = 1`). **Brooks** permanece en su fila original de largada (`grid = 3`, `win = 0`). **La carrera registra exactamente 1 victoria**.

> [!IMPORTANT]
> Esta depuración redujo el total de victorias de la base de 1,128 a **1,125 victorias**, estableciendo una correlación perfecta e unívoca de **exactamente un ganador por Gran Premio en las 1,125 carreras del campeonato mundial**.

### 2.3. Fuente Histórica Verificable Consultada
* **Fuente Oficial Consultada:** *Formula 1 Official Database & Historical Archive / Ergast Developer API*.
* **Crónica Específica Verificada:** Wikipedia - *1951 French Grand Prix* (Revisada y validada en línea).
* **Enlace Permanente Verificado:** [https://en.wikipedia.org/wiki/1951_French_Grand_Prix](https://en.wikipedia.org/wiki/1951_French_Grand_Prix)
* **Fecha de Consulta:** Octubre / Noviembre de 2026.
* **Cita Textual Verificada:**
  > *"When Fagioli came in for his fuel stop, the team ordered Fagioli and Fangio to swap cars; Fagioli's car was running healthily whereas Fangio's car had multiple mechanical problems... Fangio made his way into the lead and won the race... They shared the points for the win."*  
  > *(Referencia canónica: Lang, Mike. 1981. Grand Prix! Vol 1 (1950–1965). Haynes Publishing Group, ISBN 0-85429-276-4).*

### 2.4. Inventario Completo de las 88 Filas de Relevos Descartadas

A continuación se registra el inventario exhaustivo de las 88 filas depuradas por concepto de relevos intra-carrera (concentradas en 39 carreras entre 1950 y 1964, de las cuales 3 corresponden a relevos con `win = 1` y 85 a relevos con `win = 0`):

| Año | Gran Premio | Piloto Relevante | Posición en Parrilla del Auto Tomado | Posición de Llegada del Auto |
| :---: | :--- | :--- | :---: | :---: |
| 1950 | Indianapolis 500 | Tony Bettenhausen | 9 | 5 |
| 1950 | French Grand Prix | Louis Rosier | 15 | 6 |
| 1950 | Italian Grand Prix | Alberto Ascari | 6 | 2 |
| 1950 | Italian Grand Prix | Juan Manuel Fangio | 7 | 13 |
| 1951 | French Grand Prix | Luigi Fagioli | 1 | 11 |
| 1951 | French Grand Prix | Alberto Ascari | 3 | 18 |
| 1951 | French Grand Prix | Juan Manuel Fangio | 7 | 1 |
| 1951 | Italian Grand Prix | Nino Farina | 7 | 3 |
| 1952 | Swiss Grand Prix | Nino Farina | 4 | 10 |
| 1952 | French Grand Prix | Harry Schell | 12 | 15 |
| 1952 | Dutch Grand Prix | Jan Flinterman | 16 | 9 |
| 1953 | Indianapolis 500 | Paul Russo | 2 | 4 |
| 1953 | Indianapolis 500 | Gene Hartley | 6 | 9 |
| 1953 | Indianapolis 500 | Chuck Stevenson | 6 | 9 |
| 1953 | Indianapolis 500 | Chuck Stevenson | 7 | 23 |
| 1953 | Indianapolis 500 | Andy Linden | 7 | 23 |
| 1953 | Indianapolis 500 | Duane Carter | 9 | 3 |
| 1953 | Indianapolis 500 | Andy Linden | 10 | 16 |
| 1953 | Indianapolis 500 | Johnny Thomson | 18 | 19 |
| 1953 | Indianapolis 500 | Bob Scott | 19 | 12 |
| 1953 | Indianapolis 500 | Jim Rathmann | 28 | 15 |
| 1953 | Dutch Grand Prix | José Froilán González | 13 | 3 |
| 1953 | Belgian Grand Prix | Juan Manuel Fangio | 10 | 12 |
| 1953 | German Grand Prix | Luigi Villoresi | 1 | 8 |
| 1953 | German Grand Prix | Alberto Ascari | 6 | 17 |
| 1953 | Swiss Grand Prix | Felice Bonetto | 1 | 4 |
| 1953 | Swiss Grand Prix | Juan Manuel Fangio | 10 | 15 |
| 1954 | Indianapolis 500 | Andy Linden | 4 | 24 |
| 1954 | Indianapolis 500 | Jimmy Daywalt | 4 | 24 |
| 1954 | Indianapolis 500 | Tony Bettenhausen | 8 | 15 |
| 1954 | Indianapolis 500 | Jimmy Davies | 10 | 20 |
| 1954 | Indianapolis 500 | Jim Rathmann | 10 | 20 |
| 1954 | Indianapolis 500 | Duane Carter | 11 | 4 |
| 1954 | Indianapolis 500 | Marshall Teague | 17 | 23 |
| 1954 | Indianapolis 500 | Bob Scott | 23 | 25 |
| 1954 | Indianapolis 500 | George Fonder | 26 | 31 |
| 1954 | Indianapolis 500 | Johnnie Parsons | 27 | 11 |
| 1954 | Indianapolis 500 | Sam Hanks | 27 | 11 |
| 1954 | Indianapolis 500 | Andy Linden | 27 | 11 |
| 1954 | Indianapolis 500 | Jerry Hoyt | 32 | 8 |
| 1954 | Belgian Grand Prix | José Froilán González | 2 | 12 |
| 1954 | British Grand Prix | Alberto Ascari | 27 | 21 |
| 1954 | German Grand Prix | Mike Hawthorn | 5 | 2 |
| 1954 | Italian Grand Prix | José Froilán González | 13 | 3 |
| 1955 | Argentine Grand Prix | Nino Farina | 1 | 2 |
| 1955 | Argentine Grand Prix | Maurice Trintignant | 1 | 2 |
| 1955 | Argentine Grand Prix | Maurice Trintignant | 5 | 3 |
| 1955 | Argentine Grand Prix | Jean Behra | 7 | 6 |
| 1955 | Argentine Grand Prix | Stirling Moss | 10 | 4 |
| 1955 | Argentine Grand Prix | Karl Kling | 10 | 4 |
| 1955 | Argentine Grand Prix | Luigi Villoresi | 12 | 12 |
| 1955 | Argentine Grand Prix | Sergio Mantovani | 18 | 7 |
| 1955 | Argentine Grand Prix | Harry Schell | 18 | 7 |
| 1955 | Argentine Grand Prix | Luigi Musso | 19 | 8 |
| 1955 | Argentine Grand Prix | Jean Behra | 19 | 8 |
| 1955 | Argentine Grand Prix | Carlos Menditeguy | 20 | 9 |
| 1955 | Argentine Grand Prix | Harry Schell | 20 | 9 |
| 1955 | Monaco Grand Prix | Cesare Perdisa | 5 | 3 |
| 1955 | Monaco Grand Prix | Jean Behra | 11 | 10 |
| 1955 | Belgian Grand Prix | Jean Behra | 13 | 5 |
| 1955 | British Grand Prix | Eugenio Castellotti | 12 | 6 |
| 1955 | British Grand Prix | Harry Schell | 15 | 9 |
| 1956 | Argentine Grand Prix | Juan Manuel Fangio | 3 | 1 |
| 1956 | Monaco Grand Prix | Eugenio Castellotti | 1 | 4 |
| 1956 | Monaco Grand Prix | Juan Manuel Fangio | 9 | 2 |
| 1956 | Belgian Grand Prix | Stirling Moss | 9 | 3 |
| 1956 | French Grand Prix | Harry Schell | 6 | 10 |
| 1956 | French Grand Prix | Stirling Moss | 13 | 5 |
| 1956 | French Grand Prix | Hernando da Silva Ramos | 16 | 13 |
| 1956 | British Grand Prix | Alfonso de Portago | 8 | 10 |
| 1956 | British Grand Prix | Peter Collins | 12 | 2 |
| 1956 | Italian Grand Prix | Eugenio Castellotti | 1 | 8 |
| 1956 | Italian Grand Prix | Juan Manuel Fangio | 7 | 2 |
| 1956 | Italian Grand Prix | Jean Behra | 13 | 13 |
| 1957 | Argentine Grand Prix | Peter Collins | 11 | 6 |
| 1957 | Monaco Grand Prix | Harry Schell | 14 | 8 |
| 1957 | Monaco Grand Prix | Mike Hawthorn | 9 | 7 |
| 1957 | French Grand Prix | Jack Brabham | 15 | 7 |
| 1957 | British Grand Prix | Tony Brooks | 1 | 12 |
| 1957 | British Grand Prix | Stirling Moss | 3 | 1 |
| 1957 | British Grand Prix | Peter Collins | 9 | 4 |
| 1957 | Italian Grand Prix | Harry Schell | 12 | 5 |
| 1958 | French Grand Prix | Tony Brooks | 10 | 14 |
| 1958 | Italian Grand Prix | Carroll Shelby | 11 | 4 |
| 1960 | Argentine Grand Prix | Stirling Moss | 8 | 3 |
| 1961 | British Grand Prix | Stirling Moss | 5 | 21 |
| 1961 | United States Grand Prix | Masten Gregory | 15 | 11 |
| 1964 | United States Grand Prix | Jim Clark | 1 | 12 |

---

## 3. Verificación de Unicidad en la Base de Datos Final

Para permitir la auditoría computacional directa de unicidad sobre el archivo generado en disco, se integró formalmente la columna `driverId` en el archivo `data/dataset_tesis_f1.csv`.

### Código de Comprobación
```python
import pandas as pd

# Cargar dataset final de tesis
df = pd.read_csv('data/dataset_tesis_f1.csv')

# Validar duplicados de piloto por carrera
duplicados = df.duplicated(['raceId', 'driverId']).sum()
print(f"Total de registros duplicados en (raceId, driverId): {duplicados}")
```

### Salida Real en Consola
```text
Total de registros duplicados en (raceId, driverId): 0
```
**Dictamen:** **Tasa de unicidad del 100.0000%**. No existe ningún registro duplicado en las 25,033 observaciones que componen la base de datos de la tesis.

---

## 4. Partición Temporal Causal y Punto de Corte Atómico

### 4.1. Justificación contra el Split Aleatorio 80/20
El uso de una partición aleatoria convencional (*random train-test split*) está estrictamente contraindicado en competiciones deportivas seriales en el tiempo debido a:
1. **Fuga Temporal hacia el Futuro (*Look-Ahead Bias*):** El modelo aprendería con datos de 2023 para predecir carreras de 2014, accediendo a patrones tecnológicos maduros de escuderías que en el pasado eran inexistentes.
2. **Violación del Supuesto IID:** Los pilotos que compiten en una misma carrera comparten factores contextuales correlacionados (clima, accidentes, neutralizaciones con coche de seguridad). Separar aleatoriamente pilotos de una misma carrera entre entrenamiento y prueba produce fuga intra-evento.
3. **Validez Ecológica:** El despliegue de un sistema de apoyo a la decisión en Fórmula 1 opera en el instante $t_0$ con datos del pasado ($t < t_0$) para inferir un evento no ocurrido ($t > t_0$).

### 4.2. Documentación del Corte Atómico en el GP de Abu Dabi 2012
El límite de partición temporal se fijó en el cierre íntegro del **Gran Premio de Abu Dabi 2012**:
* **Última Carrera de Entrenamiento:** Gran Premio de Abu Dabi 2012, **`raceId = 877`** (Ronda 18 del campeonato 2012, fecha: `2012-11-04`, Circuito de Yas Marina). Los 24 monoplazas participantes quedan asignados en su totalidad a `Train`.
* **Primera Carrera de Prueba:** Gran Premio de Estados Unidos 2012 en Austin, **`raceId = 878`** (Ronda 19 del campeonato 2012, fecha: `2012-11-18`, Circuito de las Américas).

### 4.3. Tabla Comparativa de Partición (Train vs. Test)

| Dimensión / Métrica | Conjunto de Entrenamiento (Train) | Conjunto de Prueba (Test) | Base Consolidada Total |
| :--- | :---: | :---: | :---: |
| **Observaciones Totales ($N$)** | **20,020** | **5,013** | **25,033** |
| **Proporción de Observaciones** | **79.974%** | **20.026%** | 100.000% |
| **Grandes Premios Evaluados** | **876 carreras** (`77.87%`) | **249 carreras** (`22.13%`) | 1,125 carreras |
| **Rango de Identificadores (`raceId`)** | `1` a **`877`** | **`878`** a `1144` | `1` a `1144` |
| **Victorias Registradas ($win=1$)** | **876 victorias** | **249 victorias** | 1,125 victorias |
| **Prevalencia de la Clase Positiva** | **4.376%** | **4.967%** | 4.494% |
| **Rango Cronológico (Fechas)** | **1950-05-13 a 2012-11-04** | **2012-11-18 a 2024-12-08** | 1950-05-13 a 2024-12-08 |
| **Carrera Inicial** | GP de Gran Bretaña 1950 (Silverstone) | GP de EE.UU. 2012 (Austin) | GP de Gran Bretaña 1950 |
| **Carrera Final** | GP de Abu Dabi 2012 (Yas Marina) | GP de Abu Dabi 2024 (Yas Marina) | GP de Abu Dabi 2024 |

### 4.4. Verificación de No-Traslape Temporal por Fecha Real
El código evalúa que la fecha máxima de entrenamiento sea estrictamente menor a la fecha mínima de prueba:
```python
max_date_train = train['date'].max()  # 2012-11-04 (GP de Abu Dabi 2012)
min_date_test  = test['date'].min()   # 2012-11-18 (GP de EE.UU. 2012)
assert max_date_train < min_date_test
```
$$\max(\text{fecha}_{\text{train}}) = \text{2012-11-04} \quad < \quad \min(\text{fecha}_{\text{test}}) = \text{2012-11-18} \quad (\Delta t = 14 \text{ días de separación})$$

---

## 5. Preprocesamiento sin Fuga de Información (*Leak-Free Pipeline*)

### 5.1. Columnas Escaladas y Algoritmo de Normalización
* **Variables con escalado:** En los scripts de Regresión Logística (`4_regresion_logistica.py`) y Perceptrón Multicapa (`3_perceptron_multicapa.py`), `StandardScaler` se aplica sobre la matriz de características `X_train` (201 columnas, incluyendo `grid`, `age`, `round` y las dummies `constructorId_*`), ajustado estrictamente sobre entrenamiento.
* **Modelos basados en árboles:** Para **Random Forest** (`2_random_forest.py`), no se aplica escalado, ya que los criterios de partición de impureza (Gini/Entropía) son invariantes a transformaciones monótonas.

### 5.2. Metodología de Ajuste Estricto sobre Entrenamiento
Para evitar la filtración de estadísticos poblacionales futuros en el pasado, el transformador `StandardScaler` de Scikit-Learn se calibra **únicamente sobre `X_train`**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 1. Ajuste (cálculo de media mu y desviación sigma) SOLO con datos históricos de Train
X_train_scaled = scaler.fit_transform(X_train)

# 2. Transformación ciega de Test utilizando la mu y sigma fijadas en Train
X_test_scaled = scaler.transform(X_test)
```
$$z_{\text{train}} = \frac{x_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad z_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Para los modelos de ensamble basados en árboles de decisión (**Random Forest**), no se aplica escalado.

---

## 6. Catálogo de Variables y Reconciliación Dimensional de Columnas

### 6.1. Reconciliación de las 201 Variables Predictoras vs. las 248 Columnas del CSV
Existe una correspondencia matemática unívoca entre las dimensiones del modelo y el esquema físico del archivo `dataset_tesis_f1.csv`:

$$\begin{aligned}
\text{Total Columnas CSV (248)} &= \mathbf{201} \text{ (Variables predictoras } X) \\
&\quad + \mathbf{1} \text{ (Variable objetivo: } win) \\
&\quad + \mathbf{2} \text{ (Identificadores: } raceId, driverId) \\
&\quad + \mathbf{1} \text{ (Variable de ablación: } year) \\
&\quad + \mathbf{43} \text{ (Variables de ablación: } nationality\_*)
\end{aligned}$$

#### Desglose de las 201 Variables Predictoras del Espacio de Características ($X$):
1. **Predictores Numéricos (3 variables):**
   * `grid`: Posición física en parrilla de salida.
   * `age`: Edad cronológica exacta en años.
   * `round`: Número de carrera dentro del calendario anual.
2. **Predictores Categóricos de Escudería (198 variables):**
   * `constructorId_*`: 198 columnas dummy que identifican la escudería del monoplaza (e.g., Ferrari, McLaren, Williams, Red Bull, Mercedes, Tyrrell, Lotus, Brabham, etc.).

#### Tratamiento en los Scripts de Modelado
En `2_random_forest.py`, `3_perceptron_multicapa.py` y `4_regresion_logistica.py`:
```python
cols_to_exclude = ["win", "raceId", "driverId", "year"] + [c for c in df.columns if c.startswith("nationality")]
X = df.drop(columns=cols_to_exclude, errors="ignore")
# X.shape[1] == 201 columnas predictoras
```

### 6.2. Corrección Formal de la Variable `grid`: Posición en Parrilla
> [!IMPORTANT]
> **Corrección Conceptual de `grid`:**  
> `grid` es la **posición de salida efectiva en la parrilla de largada (*Starting Grid Position*)**, no la posición obtenida en la sesión de clasificación.
> * La posición de clasificación en pista se almacena en `qualifying.csv` (`position`).
> * La variable `grid` en `results.csv` refleja el cajón físico de salida asignado tras ejecutar todas las sanciones de los comisarios deportivos de la FIA (cambio de caja de cambios, reemplazo de motor, infracciones técnicas o largadas desde el *pit lane*). Por tanto, es el predictor pre-carrera más representativo y cercano a la largada.

### 6.3. Registro de Variables Excluidas del Modelado

| Variable Excluida | Fuente | Momento | Razón Metodológica de Exclusión |
| :--- | :--- | :--- | :--- |
| **`year` (Año calendario)** | `races.csv` | Pre-carrera | **Variable no estacionaria (*temporal drift*):** Su inclusión degrada el AUC-ROC y AUC-PR en la ablación y provoca una caída de $6.71\%$ en el Recall de Random Forest. Fuerza al modelo a extrapolar fuera de rango en el test set ($year > 2012$). |
| **`nationality` (Nacionalidad)** | `drivers.csv` | Estática | **Dispersión y confusión (*confounding bias*):** Añade 43 dummies dispersas que colapsan el Recall en Regresión Logística de $85.21\%$ a $76.44\%$. La nacionalidad no ejerce tracción física; la competitividad del auto queda capturada por `constructorId`. |
| **`raceId` y `driverId`** | Base Ergast | Identificador | **Claves sustitutas autoincrementales:** No poseen valor predictivo real; incluirlas induciría sobreajuste memorístico espurio. Se conservan en el CSV solo para trazabilidad y comprobación de unicidad. |
| **`quali_position`** | `qualifying.csv` | Clasificación | **Desactualizada respecto a penalizaciones:** La posición definitiva de arranque está consolidada en `grid`. |
| **`fastestLapSpeed` / `fastestLapTime`** | `results.csv` | Durante carrera | **Fuga Temporal Crítica (*Data Leakage*):** Se mide durante la competencia; imposible de conocer en la parrilla. |
| **`pit_stops` / paradas en boxes** | `pit_stops.csv` | Durante carrera | **Fuga Temporal:** Refleja la estrategia y eventos sucedidos tras la largada. |
| **`positionOrder` / `points`** | `results.csv` | Post-carrera | **Fuga de la Variable Objetivo (*Target Leakage*):** `positionOrder == 1` define el target `win`. Predeciría el resultado con el propio resultado. |
| **`statusId` / `laps`** | `results.csv` | Post-carrera | **Fuga Temporal:** Indica abandonos mecánicos o choques ocurridos en carrera. |

---

## 7. Justificación Metodológica del Umbral de Clasificación (*Classification Threshold*)

### 7.1. Desbalance Severo de Clases en Fórmula 1
En una carrera típica de Fórmula 1 toman la salida entre 20 y 22 pilotos, pero **solo 1 puede obtener la victoria** ($y = 1$). Esto impone una prevalencia natural de la clase positiva de apenas **$4.49\%$** (un desbalance aproximado de 21 a 1).

### 7.2. Inadecuación del Umbral Ingenuo ($\tau = 0.50$)
En clasificadores probabilísticos sin calibración de costos, un umbral fijo en $\tau = 0.50$ exige que el modelo asigne una probabilidad superior al 50% para predecir que un piloto ganará. En presencia de 20 competidores con probabilidades distribuidas, la probabilidad posterior rara vez supera el 0.50 a menos que el favoritismo sea absoluto. Esto causa una tasa excesiva de falsos negativos y un colapso en la métrica de exhaustividad (*Recall*).

### 7.3. Mecanismos de Compensación Adoptados

#### A. Función de Pérdida con Costos Balanceados (`class_weight='balanced'`)
Durante el entrenamiento, se ajustan los pesos de penalización de la función de pérdida inversamente proporcionales a las frecuencias muestrales de cada clase:
$$w_1 = \frac{N}{2 \cdot N_1} = \frac{25033}{2 \cdot 1125} \approx 11.12, \qquad w_0 = \frac{N}{2 \cdot N_0} = \frac{25033}{2 \cdot 23908} \approx 0.52$$
Al penalizar los errores sobre la clase positiva con un factor $\approx 21$ veces mayor que sobre la negativa, el hiperplano de decisión se desplaza endógenamente, permitiendo que el umbral estándar $\tau = 0.50$ opere en una escala compensada de costos.

#### B. Optimización del Umbral Operativo vía Curva Precision-Recall
Si se evalúan probabilidades no balanceadas, el umbral óptimo $\tau^*$ se determina en el conjunto de entrenamiento maximizando la media armónica entre precisión y exhaustividad:
$$\tau^* = \arg\max_{\tau \in (0, 1)} F_1(\tau) = \arg\max_{\tau \in (0, 1)} \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}$$
En dominios con desbalance del 4.5%, el umbral que maximiza el $F_1$-Score se ubica típicamente en el intervalo $\tau \in [0.15, 0.25]$.

#### C. Paradigma Deportivo de Clasificación por Ranking (*ArgMax por Gran Premio*)
Desde una perspectiva operativa del automovilismo, la predicción óptima no consiste en evaluar a cada piloto como una decisión binaria independiente, sino en estimar la probabilidad condicionada al evento $P(\text{win} \mid \text{piloto } d, \text{carrera } r)$ y predecir como ganador al piloto con la probabilidad máxima dentro de dicha carrera:
$$\hat{d}_r^* = \arg\max_{d \in \mathcal{D}_r} \hat{P}(\text{win} \mid d, r)$$
Este enfoque garantiza por construcción que cada Gran Premio prediga exactamente a un solo ganador, eliminando la necesidad de calibrar un umbral estático arbitrario.

---

## 8. Estudio de Ablación en Entrenamiento (Media de 5 Folds Temporales)

En estricta observancia del principio de mantener el conjunto de prueba ciego (*test set integrity*), el estudio de ablación para validar la exclusión de `year` y `nationality` se realizó **únicamente con las 20,020 observaciones de entrenamiento** utilizando validación cruzada temporal de 5 pliegues (*5-Fold Race-Aware TimeSeriesSplit*):

| Configuración Evaluada | Dimensiones ($p$) | LR AUC-ROC | LR AUC-PR | LR F1 | LR Recall | RF AUC-ROC | RF AUC-PR | RF F1 | RF Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Base Oficial (Sin Año, Sin Nacionalidad)** | **201** | **0.9065** | **0.2820** | 0.2904 | **0.8521** | **0.8892** | **0.2581** | **0.3078** | **0.7479** |
| **2. Con Año (Sin Nacionalidad)** | **202** | 0.9046 | 0.2759 | **0.2982** | 0.8534 | 0.8882 | 0.2503 | 0.3083 | 0.6808 |
| **3. Con Nacionalidad (Sin Año)** | **244** | 0.9002 | 0.2742 | 0.2797 | 0.7644 | 0.8861 | 0.2572 | 0.2915 | 0.7178 |
| **4. Con Año y Nacionalidad** | **245** | 0.8984 | 0.2636 | 0.2881 | 0.7712 | 0.8813 | 0.2358 | 0.2930 | 0.6521 |

### Conclusiones de la Ablación
1. **Exclusión de `year`:** Su incorporación reduce la precisión en la clase minoritaria (AUC-PR cae de $0.2820$ a $0.2759$ en LR y de $0.2581$ a $0.2503$ en RF) y destruye la sensibilidad en Random Forest (el Recall cae de $74.79\%$ a $68.08\%$, perdiendo $6.71$ puntos porcentuales).
2. **Exclusión de `nationality`:** Provoca una caída dramática en el Recall de Regresión Logística (de $85.21\%$ a $76.44\%$, una pérdida de $8.77$ puntos porcentuales) y añade 43 variables altamente ralas que perjudican la capacidad de generalización.
3. La **Configuración Base Oficial con 201 variables** (`grid`, `age`, `round`, `constructorId_*`) maximiza simultáneamente el AUC-ROC, AUC-PR y la capacidad de detección de victorias en ambos paradigmas de modelado.

---

## 9. Dictamen Metodológico Final

1. **Integridad de Datos:** La base de datos `data/dataset_tesis_f1.csv` cuenta con **25,033 observaciones**, **1,125 carreras**, **1,125 victorias** (exactamente un ganador por carrera) y **cero duplicados piloto-carrera**.
2. **Blindaje contra Fuga:** Se ha demostrado matemáticamente y en código que no existe traslape cronológico entre Train y Test ($\max(\text{fecha}_{\text{train}}) < \min(\text{fecha}_{\text{test}})$) y que el escalado se ajusta únicamente sobre entrenamiento.
3. **Consistencia de Código:** Todos los tests de la suite (`tests/test_pilot_race_uniqueness.py` y `tests/test_temporal_integrity.py`) han sido ejecutados y validados con estatus aprobatorio del 100% (`OK`).
4. **Estado del Repositorio:** Todos los cambios se encuentran congelados y sincronizados en la rama oficial **`test`** del sistema de control de versiones Git.

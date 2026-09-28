# Análisis de Variables Adicionales para Mejorar Modelos Predictivos

## Variables Propuestas para Incrementar el Rendimiento

### 1. ✓ Experiencia del Piloto en el Circuito **(RECOMENDADA - IMPERATIVA)**

**Tipo**: Numérica discreta  
**Fuente**: `races.csv` + `results.csv` (derivado)

**Descripción**: Número de carreras previas que un piloto ha corrido en un circuito específico hasta el momento de esta carrera

**Justificación Teórica**:
- Pilotos con mayor experiencia en un circuito conocen las líneas de carrera óptimas
- Mejor adaptación a condiciones meteorológicas específicas del circuito
- Reducción de errores técnicos y aprendizaje de trazado
- Circuitos técnicos (Mónaco, Suzuka) requieren mayor conocimiento del trazado

**Implementación Sugerida**:
```python
# Ordenar cronológicamente
df = df.sort_values(['year', 'date'])

# Calcular experiencia acumulada POR CIRCUITO
df['experiencia_circuito'] = df.groupby(['driverId', 'circuitId']).cumcount()
```

**Análisis Esperado**:
- Distribución esperada: 0-50 carreras (la mayoría de pilotos 0-10)
- Correlación esperada: Positiva con probabilidad de victoria
- Interpretación: 1 unidad = 1 carrera previa en ese circuito

---

### 2. Experiencia del Piloto en Constructor (Escudería)

**Tipo**: Numérica discreta  
**Fuente**: `races.csv` + `results.csv` (derivado)

**Descripción**: Número de carreras previas que un piloto ha corrido con un constructor específico

**Justificación Teórica**:
- Pilotos recién contratados necesitan adaptación a la tecnología del constructor
- Mejor entendimiento de la estrategia y mecánica del auto
- Historial previo con constructor puede influir en el rendimiento

**Implementación**:
```python
df['experiencia_constructor'] = df.groupby(['driverId', 'constructorId']).cumcount()
```

**Importancia**: Alta - Los constructores tienen características muy diferentes (aerodinámica, motor, rendimiento)

---

### 3. Ranking de Constructor en Temporada (Standings)

**Tipo**: Numérica discreta (ranking 1-20)  
**Fuente**: `constructor_standings.csv`

**Descripción**: Posición del constructor en el campeonato de constructores de esa temporada específica

**Justificación Teórica**:
- Constructores mejores tienen más recursos de ingeniería
- Influye directamente en la calidad del automóvil (motor, aerodinámica)
- Correlación alta con probabilidad de victoria (Mercedes, Ferrari típicamente top 5)

**Implementación**:
```python
# Merge con constructor_standings por raceId y constructorId
df = df.merge(constructor_standings, on=['raceId', 'constructorId'], how='left')
```

**Importancia**: Muy Alta - Variable proxy de calidad del vehículo

---

### 4. Ranking de Piloto en Temporada (Standings)

**Tipo**: Numérica discreta (ranking 1-20)  
**Fuente**: `driver_standings.csv`

**Descripción**: Posición del piloto en el campeonato de pilotos en ese punto de la temporada

**Justificación Teórica**:
- Pilotos ranking alto tienen mayor confianza y recursos del equipo
- Inversión de recursos del constructor hacia pilotos en posiciones altas del campeonato

**Implementación**:
```python
df = df.merge(driver_standings, on=['raceId', 'driverId'], how='left')
```

**Importancia**: Alta - Captura estado actual de rendimiento y recursos del piloto

---

### 5. Densidad de Carrera en Circuito (Complejidad de Trazado)

**Tipo**: Numérica continua  
**Fuente**: `circuits.csv`

**Descripción**: Métrica que combina longitud del circuito con número de curvas (complejidad del trazado)

**Justificación Teórica**:
- Circuitos más complejos requieren mayor piloto y conocimiento
- Circuitos cortos y sencillos (Monza) vs complejos (Mónaco) influyen en ventaja por experiencia

**Posibles Métricas**:
```python
df['circuito_densidad'] = df['lat'] / df['lng']  # Aproximación de complejidad
df['circuito_longitud_categoria'] = categorize(df['circuit_length'], bins=[0, 4000, 6000, 8000])
```

**Importancia**: Media - Variables contextuales del circuito pueden interactuar con experiencia

---

### 6. Historial de Abandonos vs Carreras Completadas

**Tipo**: Numérica continua (porcentaje 0-1)  
**Fuente**: `results.csv` + `status.csv`

**Descripción**: Porcentaje de carreras que el piloto no pudo completar (abandonos técnicos)

**Justificación Teórica**:
- Pilotos con alta tasa de abandonos menos confiables para predecir victoria
- Factores mecánicos pueden predecir probabilidad de abandonos
- Inversa de reliability

**Implementación**:
```python
# Agrupar por piloto año y calcular tasa de abandonos
df['historial_abandonos'] = df.groupby(['driverId', 'year'])['statusId'].transform(
    lambda x: (x != 1).mean()  # Asumiendo statusId=1 es "completado"
)
```

**Importancia**: Media - Más relevante para clasificación de "abandono" que "victoria"

---

### 7. Temperatura y Condición Meteorológica

**Tipo**: Categórica o Numérica continua  
**Fuente**: Variables climáticas en `races.csv` o API externa

**Descripción**: Temperatura de la pista, condiciones climáticas (seco, lluvioso, mixto)

**Justificación Teórica**:
- Condiciones lluviosas cambian dinámica de carrera drásticamente
- Circuitos con conocimiento en lluvia tienen ventaja

**Implementación**:
```python
# Si datos disponibles en races.csv
df['condicion_climatica'] = categorizar_clima(df['weather'])
```

**Importancia**: Alta - Pero requiere datos adicionales tal vez no disponibles en dataset original

---

### 8. Edad del Piloto al Inicio de Carrera

**Tipo**: Numérica continua  
**Fuente**: `drivers.csv` + `races.csv` (derivado)

**Descripción**: Edad del piloto específica al inicio de esa carrera en particular

**Justificación Teórica**:
- Diferente de "edad actual" que calculamos como (fecha_ahora - dob)
- Capta variación de edad en tiempo
- Pilotos experimentados (más viejos) pueden declinar fisiológicamente

**Implementación**:
```python
df['edad_inicio_carrera'] = (pd.to_datetime(df['date']) - df['dob']).dt.days / 365
```

**Importancia**: Alta - Más preciso que la variable de edad actual calculada

---

### 9. Inversión de Posiciones (Mejora desde Grid)

**Tipo**: Numérica discreta (negativa=perdió posiciones, positiva=ganó)  
**Fuente**: `results.csv`

**Descripción**: `positionOrder - grid` (diferencia entre posición inicial y final)

**Justificación Teórica**:
- Captura rendimiento y estrategia de carrera
- Pilotos que consistentemente ganan posiciones desde detrás mostraría habilidad

**Implementación**:
```python
df['mejora_posiciones'] = df['positionOrder'] - df['grid']
```

⚠️ **Problema**: Variable post-hoc (conocida después de carrera) - Solo usar para análisis histórico, no predicción a priori

**Importancia**: Muy Alta - Pero **FUENTE DE FUGA TEMPORAL** si se usa para predicción

---

### 10. Número de Victorias en Temporada (Reciente)

**Tipo**: Numérica discreta  
**Fuente**: `results.csv` + `races.csv` (agregado)

**Descripción**: Número de victorias del piloto en la temporada actual hasta esa carrera

**Justificación Teórica**:
- Pilotos en racha de victorias mayor confianza y recursos
- Momentum psicológico y técnico

**Implementación**:
```python
df['victorias_temporada'] = df.groupby(['driverId', 'year'])['win'].transform('cumsum') - 1  # -1 para no incluir carrera actual
```

**Importancia**: Muy Alta - Pero **FUENTE DE FUGA TEMPORAL** si no se excluye la carrera actual

---

## Análisis de Prioridades

### Fase 1 - Variables Críticas (Implementar Primero)

1. **✓ Experiencia del Piloto en el Circuito** - RECOMENDACIÓN del DR
2. **Experiencia del Piloto en Constructor** - Complemento muy importante
3. **Ranking de Constructor en Temporada** - Proxy de calidad del vehículo
4. **Ranking de Piloto en Temporada** - Estado actual de rendimiento

### Fase 2 - Variables de Alto Valor

5. **Edad del Piloto al Inicio de Carrera** - Reemplazo de edad actual
6. **Victorias en Temporada (hasta fecha)** - Momentum y racha

### Fase 3 - Variables Complementarias

7. **Densidad de Circuito** - Contexto del trazado
8. **Historial de Abandonos** - Reliability
9. **Condiciones Meteorológicas** - Si datos disponibles

### Fase 4 - Variables No Recomendadas

⚠️ **Mejora de Posiciones (post-hoc)** - Solo útil para análisis histórico, NO para predicción

---

## Recomendación Final

Para la tesis, recomiendo incluir en **Orden de Prioridad**:

### Mínimo Viable (Variables Imperativas):
1. Experiencia del Piloto en el Circuito
2. Edad del Piloto al Inicio de Carrera (reemplaza `age`)

### Recomendado para Mejorar Rendimiento:
3. Ranking de Constructor en Temporada
4. Ranking de Piloto en Temporada
5. Experiencia en Constructor

### Ideal para Investigación Exhaustiva:
6. Victorias en Temporada (hasta fecha)
7. Historial de Abandonos

---

## Implementación Sugerida

El script de preparación de datos debería:

1. **Crear dataset versión base** (variable actual: `win`, `age`, `grid`, etc.)
2. **Crear dataset versión mejorado** con nuevas características
3. **Facilitar comparación** de rendimiento entre versiones
4. **Documentar cada transformación** y justificación teórica

Esto permitirá:
- Evaluar empíricamente si las nuevas variables mejoran AUC, F1-score y otras métricas
- Analizar incremento de dimensionalidad vs ganancia predictive
- Documentar qué variables son más determinantes estadísticamente

---

**Generando script de preparación versión 2.0 ahora...**
(incluye `experiencia_circuito` y preparado para agregar más variables)
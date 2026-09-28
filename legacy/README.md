# Scripts Históricos / Legacy del Proyecto

Este directorio almacena scripts preliminares desarrollados en las fases iniciales del proyecto. 

## Estado Metodológico
Los scripts en este directorio han sido **sustituidos** por el pipeline oficial de la tesis debido a las siguientes razones identificadas durante la auditoría técnica interna:

1. **Uso de Partición Aleatoria**: Empleaban `train_test_split(..., shuffle=True)` en lugar de la partición temporal cronológica 80/20 requerida para datos secuenciales de Fórmula 1.
2. **Fuga Temporal (Data Leakage)**: Incluían variables que únicamente se conocen durante o después de la carrera (`fastestLapSpeed`, `milliseconds_pit_stop`, `milliseconds_lap_time`, `positionOrder`).
3. **Escalado Global**: Aplicaban transformaciones antes del split temporal.

## Archivos Archivados
- `1_preparacion_dataset.py`: Versión preliminar del generador de datasets con variables de carrera.
- `modelo_regresion_lineal.py`: Versión preliminar del modelo lineal con split aleatorio.
- `f1_win.py`: Script exploratorio original de redes neuronales y árboles.

## Pipeline Actual y Vigente
El pipeline canónico oficial del proyecto se localiza en:
- Preparación del dataset pre-carrera: [`data/preparar_datos_tesis.py`](../data/preparar_datos_tesis.py)
- Modelos predictivos: Directorio [`modelos/`](../modelos/) (`1_regresion_lineal.py`, `2_random_forest.py`, `3_perceptron_multicapa.py`, `4_regresion_logistica.py`, `5_modelo_avanzado.py`).

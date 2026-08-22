# Analítica de datos — Tecmilenio

Actividades del curso. **Gael Rodríguez Jiménez** — Ingeniería en Desarrollo de Software y Sistemas Computacionales.

---

## Actividad 2 — Modelos de regresión para estimar el desempeño académico

### ▶ Ver la actividad (se abre en el navegador, no requiere descargar nada)

**[Actividad2_Modelos_Regresion.ipynb](Actividad2/Actividad2_Modelos_Regresion.ipynb)** — notebook ejecutado, con todo el código, las salidas y las 10 gráficas ya renderizadas.

Enlaces alternativos por si el visor de GitHub tarda en cargar:

- [Abrir en Google Colab](https://colab.research.google.com/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad2/Actividad2_Modelos_Regresion.ipynb) (ejecutable)
- [Abrir en nbviewer](https://nbviewer.org/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad2/Actividad2_Modelos_Regresion.ipynb) (solo lectura)

### Dataset

| Campo | Detalle |
|---|---|
| Nombre | Student Performance Factors Dataset |
| Autor | Mosap Abdel-Ghany |
| Fuente | [Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/student-performance-factors-dataset) |
| Fecha de consulta | 17 de agosto de 2026 |
| Licencia | CC0: Public Domain |
| Volumen | 6,607 registros × 20 variables |
| Variable objetivo | `Exam_Score` (numérica continua) |

> Abdel-Ghany, M. (2025). *Student Performance Factors Dataset* [Conjunto de datos]. Kaggle.
> https://www.kaggle.com/datasets/mosapabdelghany/student-performance-factors-dataset

### Pregunta predictiva

> ¿En qué medida las características académicas, personales y escolares de los estudiantes permiten estimar su puntaje de examen mediante modelos de regresión?

### Resultados (conjunto de prueba, partición 80/20 con semilla 42)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Línea base (`DummyRegressor`, media) | 2.8235 | 3.7611 | −0.0007 |
| **Regresión lineal** | **0.4442** | **1.7994** | **0.7709** |
| Ridge (modelo alternativo) | 0.4442 | 1.7994 | 0.7709 |

Validación cruzada 5-fold sobre el modelo elegido: R² medio de **0.7167** con desviación estándar de **0.0343**.

### Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| [`Actividad2/Actividad2_Modelos_Regresion.ipynb`](Actividad2/Actividad2_Modelos_Regresion.ipynb) | Notebook ejecutado y documentado |
| [`Actividad2/StudentPerformanceFactors_original.zip`](Actividad2/StudentPerformanceFactors_original.zip) | Dataset original descargado de Kaggle |
| [`Actividad2/StudentPerformanceFactors_procesado.csv`](Actividad2/StudentPerformanceFactors_procesado.csv) | CSV procesado generado en la Actividad 1 |

### Cómo reproducirlo

1. Abrir el notebook en Google Colab con el enlace de arriba.
2. Subir al entorno `StudentPerformanceFactors_original.zip` (o el CSV procesado).
3. Ejecutar todas las celdas en orden. La semilla fija (`random_state=42`) garantiza que las métricas se reproduzcan exactamente.

### Herramientas

Python 3, pandas, NumPy, scikit-learn (`Pipeline`, `ColumnTransformer`, `LinearRegression`, `Ridge`, `DummyRegressor`, `KFold`), matplotlib y seaborn.

# Analítica de datos — Tecmilenio

Actividades del curso. **Gael Rodríguez Jiménez** — Ingeniería en Desarrollo de Software y Sistemas Computacionales.

Todos los notebooks están **ejecutados**, con las salidas y las gráficas ya renderizadas: se leen completos en el navegador sin descargar ni ejecutar nada.

| Actividad | Tema | Ver |
|---|---|---|
| **4** | Clasificación con Máquinas de Soporte Vectorial (SVM) | [Abrir notebook](Actividad4/Actividad4_SVM_Wine.ipynb) |
| **3** | Clasificación con SVM y agrupamiento con K-Means | [Abrir notebook](Actividad3/Actividad3_SVM_Clustering.ipynb) |
| **2** | Modelos de regresión para estimar el desempeño académico | [Abrir notebook](Actividad2/Actividad2_Modelos_Regresion.ipynb) |

---

## Actividad 4 — Clasificación con Máquinas de Soporte Vectorial (SVM)

### ▶ Ver la actividad (se abre en el navegador, no requiere descargar nada)

**[Actividad4_SVM_Wine.ipynb](Actividad4/Actividad4_SVM_Wine.ipynb)** — notebook ejecutado, con todo el código, las salidas y las 4 gráficas ya renderizadas.

Enlaces alternativos por si el visor de GitHub tarda en cargar:

- [Abrir en Google Colab](https://colab.research.google.com/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad4/Actividad4_SVM_Wine.ipynb) (ejecutable)
- [Abrir en nbviewer](https://nbviewer.org/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad4/Actividad4_SVM_Wine.ipynb) (solo lectura)

### Dataset

| Campo | Detalle |
|---|---|
| Nombre | Wine Recognition Data |
| Autores | M. Forina *et al.*, Institute of Pharmaceutical and Food Analysis and Technologies (Génova, Italia) |
| Fuente | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine), vía [`sklearn.datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| Fecha de consulta | 9 de septiembre de 2026 |
| Licencia | CC BY 4.0 |
| Volumen | 178 registros × 13 variables numéricas |
| Variable objetivo | Variedad de vino: class_0 (59), class_1 (71), class_2 (48) |

> Aeberhard, S. y Forina, M. (1992). *Wine* [Conjunto de datos]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5PC7J

El conjunto viene incluido en scikit-learn, por lo que **no requiere descarga**: `load_wine()` lo lee de la propia instalación. Cada registro es el análisis químico de un vino italiano —contenido de alcohol, flavonoides, intensidad de color, prolina, entre otros— procedente de tres cultivares distintos de la misma región.

### La pregunta del análisis

> ¿Puede una SVM separar dos variedades de vino usando solo dos mediciones químicas? ¿Cuánto se gana al pasar de una frontera recta a una curva, y cuándo esa curva deja de ayudar y empieza a memorizar?

El ejercicio arranca deliberadamente con un subconjunto reducido —**clases 0 y 1, variables `alcohol` e `color_intensity`**, 130 registros— porque en dos dimensiones la frontera de decisión se puede dibujar y, por tanto, discutir. La extensión final repite el análisis con las 13 variables y las 3 clases.

### Resultados con 2 variables y 2 clases (prueba, partición 80/20 estratificada, semilla 42)

| Modelo SVM | Kernel | Accuracy | Precision | Recall | F1-score | Vectores de soporte |
|---|---|---|---|---|---|---|
| Modelo 1 (`C=1`) | Lineal | 0.8846 | 0.9231 | 0.8571 | 0.8889 | 23 |
| **Modelo 2** (`C=1`, `gamma='scale'`) | **RBF** | **0.9231** | **1.0000** | 0.8571 | **0.9231** | 29 |

El modelo lineal falla en 3 de los 26 casos de prueba (matriz de confusión `[[11, 1], [2, 12]]`), todos ellos en la franja de traslape entre ambas variedades. El RBF elimina el falso positivo y alcanza precisión perfecta.

Más revelador que el accuracy es la **brecha entre entrenamiento y prueba**, que mide cuánto está memorizando cada modelo:

| Modelo | Accuracy entrenamiento | Accuracy prueba | Brecha |
|---|---|---|---|
| Lineal | 0.9231 | 0.8846 | 0.0385 |
| **RBF** | 0.9327 | **0.9231** | **0.0096** |

### Efecto de los hiperparámetros

**`C`** (penalización del error) recorrido sobre `[0.01, 0.1, 1, 10, 100]` con kernel lineal: el accuracy de prueba se mantiene **constante en 0.8846** en todos los casos. Lo que cambia es el número de vectores de soporte, que cae de 75 a 18 conforme `C` crece: el modelo se vuelve menos tolerante, pero la limitación no era la penalización sino la forma recta de la frontera.

**`gamma`** (alcance de la influencia de cada punto) con kernel RBF y `C=1`:

| gamma | Accuracy entrenamiento | Accuracy prueba | Brecha |
|---|---|---|---|
| 0.01 | 0.9327 | 0.8846 | 0.0481 |
| **0.10** | 0.9231 | **0.9231** | **0.0000** |
| 1.00 | 0.9423 | 0.9231 | 0.0192 |
| 10.00 | **0.9712** | 0.8846 | 0.0865 |

`gamma=10` es el caso de libro de texto: el mejor accuracy de entrenamiento de toda la tabla y, simultáneamente, la peor generalización. La frontera se retuerce hasta rodear puntos individuales.

### Extensión: 13 variables y 3 clases

Al usar el dataset completo (142 registros de entrenamiento, 36 de prueba, métricas promediadas con `macro`), **la elección del kernel se invierte**:

| Configuración | Accuracy entrenamiento | Accuracy prueba | F1 macro |
|---|---|---|---|
| **Lineal `C=0.1`** | 0.9930 | **0.9722** | **0.9714** |
| RBF `C=1`, `gamma=0.1` | 1.0000 | 0.9722 | 0.9714 |
| Lineal `C=1` | 1.0000 | 0.9444 | 0.9447 |
| RBF `C=1`, `gamma=1` | 1.0000 | 0.5556 | 0.4694 |

El kernel lineal con regularización fuerte gana: **0.9722 de accuracy con un solo error en 36 casos**. El RBF con `gamma=1`, que era una de las mejores configuraciones en dos dimensiones, se desploma a 0.5556 —memoriza el entrenamiento por completo (accuracy 1.0) y no generaliza nada. En 13 dimensiones ya estandarizadas, ese `gamma` es demasiado local.

La conclusión práctica del ejercicio es esa: **el kernel óptimo no es una propiedad del algoritmo sino del espacio de características**, y cambiar el número de variables puede invertir por completo cuál conviene.

### Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| [`Actividad4/Actividad4_SVM_Wine.ipynb`](Actividad4/Actividad4_SVM_Wine.ipynb) | Notebook ejecutado y documentado |
| [`Actividad4/wine_dataset.csv`](Actividad4/wine_dataset.csv) | El dataset exportado a CSV, como anexo |

### Cómo reproducirlo

1. Abrir el notebook en Google Colab con el enlace de arriba.
2. Ejecutar todas las celdas en orden. **No hay que subir ningún archivo**: el dataset viene incluido en scikit-learn.
3. La semilla fija (`random_state=42`) garantiza que todas las métricas se reproduzcan exactamente.

### Herramientas

Python 3, pandas, NumPy, scikit-learn (`load_wine`, `train_test_split`, `StandardScaler`, `SVC`, métricas de clasificación) y matplotlib.

---

## Actividad 3 — Clasificación con SVM y agrupamiento con K-Means

**[Actividad3_SVM_Clustering.ipynb](Actividad3/Actividad3_SVM_Clustering.ipynb)** — notebook ejecutado.

- [Abrir en Google Colab](https://colab.research.google.com/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad3/Actividad3_SVM_Clustering.ipynb)
- [Abrir en nbviewer](https://nbviewer.org/github/Maade0n/analitica-datos-tecmilenio/blob/main/Actividad3/Actividad3_SVM_Clustering.ipynb)

Dataset: Breast Cancer Wisconsin (Diagnostic), cargado con `load_breast_cancer` de scikit-learn.

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

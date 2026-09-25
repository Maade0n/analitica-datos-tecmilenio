"""Construcción del modelo de vinos y exportación a PKL con Joblib.

Entrena un Pipeline (StandardScaler + SVC lineal) con 4 variables del dataset
Wine de scikit-learn, lo evalúa y lo guarda en modelo_wine.pkl.

Uso:
    python entrenar_modelo.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Carpeta donde vive este script: el .pkl se guarda junto a él.
CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_wine.pkl"

# Etiquetas didácticas para las 3 clases del dataset (0, 1, 2).
ETIQUETAS = {0: "Cabernet", 1: "Merlot", 2: "Pinot Noir"}
CLASES = list(ETIQUETAS.values())

# Las 4 características con las que se entrena el modelo.
VARIABLES = ["alcohol", "malic_acid", "color_intensity", "proline"]


def cargar_datos() -> pd.DataFrame:
    """Carga Wine en un DataFrame y agrega el nombre del tipo de vino."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    df["tipo_vino"] = df["target"].map(ETIQUETAS)
    return df


def main() -> None:
    df = cargar_datos()
    print("Dimensiones:", df.shape, "| faltantes:", int(df.isna().sum().sum()))
    print("Registros por clase:", df["tipo_vino"].value_counts().reindex(CLASES).to_dict())

    # X = entradas (4 variables), y = respuesta (nombre del vino).
    X = df[VARIABLES]
    y = df["tipo_vino"]

    # 80 % entrenamiento / 20 % prueba, conservando la proporción de clases.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)

    # El Pipeline une el escalado y el clasificador en un solo objeto.
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear")),
    ])
    modelo.fit(X_train, y_train)

    # Evaluación con datos que el modelo no vio al entrenar.
    y_pred = modelo.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=CLASES, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=CLASES), index=CLASES, columns=CLASES))

    # Guardar el Pipeline completo (escalador + SVM) en un archivo .pkl.
    joblib.dump(modelo, RUTA_MODELO)
    print(f"\nModelo guardado en: {RUTA_MODELO} ({RUTA_MODELO.stat().st_size} bytes)")

    # Comprobar que el modelo recuperado predice exactamente lo mismo.
    recuperado = joblib.load(RUTA_MODELO)
    assert np.array_equal(y_pred, recuperado.predict(X_test))
    print("Verificación: el modelo cargado da las mismas predicciones.")

    nuevo_vino = pd.DataFrame([[13.5, 1.8, 5.2, 1000]], columns=VARIABLES)
    print("Predicción de un vino nuevo:", recuperado.predict(nuevo_vino)[0])


if __name__ == "__main__":
    main()

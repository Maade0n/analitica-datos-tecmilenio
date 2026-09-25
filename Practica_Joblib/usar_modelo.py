"""Uso del modelo de vinos guardado en modelo_wine.pkl (sin volver a entrenar).

Uso:
    python usar_modelo.py                         -> predice los vinos de ejemplo
    python usar_modelo.py 12.37 1.17 1.95 520     -> predice un vino propio
      (orden: alcohol malic_acid color_intensity proline)
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_wine.pkl"
VARIABLES = ["alcohol", "malic_acid", "color_intensity", "proline"]


def main() -> None:
    if not RUTA_MODELO.is_file():
        sys.exit("No se encontró modelo_wine.pkl. Ejecuta primero: python entrenar_modelo.py")

    # Solo cargar .pkl de fuentes confiables: joblib.load puede ejecutar código.
    modelo = joblib.load(RUTA_MODELO)

    # Explorar el modelo recuperado.
    print("Objeto recuperado:", type(modelo).__name__)
    for nombre, paso in modelo.steps:
        print(f"  {nombre} -> {type(paso).__name__}")
    print("Variables esperadas:", list(modelo.feature_names_in_))
    print("Clases:", list(modelo.classes_))

    if len(sys.argv) == 5:
        # Un vino dado por el usuario en la línea de comandos.
        vinos = pd.DataFrame([[float(v) for v in sys.argv[1:]]], columns=VARIABLES)
    else:
        # Vinos de ejemplo de la práctica.
        vinos = pd.DataFrame([
            [14.23, 1.71, 5.64, 1065],
            [12.37, 1.17, 1.95, 520],
            [13.40, 3.91, 7.30, 750],
        ], columns=VARIABLES)

    # Una predicción por fila; no hace falta volver a entrenar.
    resultado = vinos.copy()
    resultado["vino_predicho"] = modelo.predict(vinos)
    print("\nPredicciones:")
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()

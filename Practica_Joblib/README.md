# Práctica Joblib — Modelo de vinos en PKL

**Gael Rodríguez Jiménez** · Analítica de datos, Tecmilenio · 24 de septiembre de 2026

Se entrena un `Pipeline` (`StandardScaler` + `SVC` lineal) con 4 variables del dataset Wine
(`alcohol`, `malic_acid`, `color_intensity`, `proline`), se guarda con **Joblib** en
`modelo_wine.pkl` y después se reutiliza para predecir **sin volver a entrenar**.

## Archivos

| Archivo | Qué es |
|---|---|
| [Construccion_modelo_vinos_PKL.ipynb](Construccion_modelo_vinos_PKL.ipynb) | Notebook ejecutado: exploración, entrenamiento, evaluación y exportación a PKL |
| [Practica_Uso_Modelo_Joblib.ipynb](Practica_Uso_Modelo_Joblib.ipynb) | Notebook ejecutado: carga del PKL y predicciones |
| [entrenar_modelo.py](entrenar_modelo.py) | Script que entrena el modelo y genera `modelo_wine.pkl` |
| [usar_modelo.py](usar_modelo.py) | Script que carga `modelo_wine.pkl` y predice |
| [modelo_wine.pkl](modelo_wine.pkl) | Modelo entrenado (scikit-learn 1.9.0) |

Abrir en Colab:
[construcción](https://colab.research.google.com/github/Maade0n/analitica-datos-tecmilenio/blob/main/Practica_Joblib/Construccion_modelo_vinos_PKL.ipynb) ·
[uso](https://colab.research.google.com/github/Maade0n/analitica-datos-tecmilenio/blob/main/Practica_Joblib/Practica_Uso_Modelo_Joblib.ipynb)

## Resultados

- Accuracy en prueba (36 vinos, 20 % estratificado): **91.67 %** (33 aciertos, 3 errores).
- Cabernet se clasifica sin errores; la confusión está entre Merlot y Pinot Noir.
- Vinos de ejemplo: `[14.23, 1.71, 5.64, 1065]` → Cabernet · `[12.37, 1.17, 1.95, 520]` → Merlot · `[13.40, 3.91, 7.30, 750]` → Pinot Noir.

## Cómo ejecutarlo en tu computadora

```bash
pip install -r requirements.txt
python entrenar_modelo.py              # entrena y genera modelo_wine.pkl
python usar_modelo.py                  # predice los 3 vinos de ejemplo
python usar_modelo.py 12.37 1.17 1.95 520   # predice un vino propio
```

> El `.pkl` debe cargarse con **scikit-learn 1.9.0**, la misma versión con la que se guardó.
> Carga solo archivos `.pkl` de fuentes confiables: `joblib.load` puede ejecutar código.

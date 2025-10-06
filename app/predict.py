# app/predict.py
"""
Módulo para cargar el modelo y realizar predicciones de cáncer de mama.
"""

import os
import logging
import numpy as np
import joblib

# Configuración de logging
logger = logging.getLogger(__name__)

# Ruta del modelo
MODEL_PATH = "model/model.pkl"

# Cargar modelo al inicio (una sola vez)
if not os.path.exists(MODEL_PATH):
    logger.error(f"Modelo no encontrado en {MODEL_PATH}")
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

try:
    modelo = joblib.load(MODEL_PATH)
    logger.info("✅ Modelo cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    raise


def make_prediction(features_list):
    """
    Realiza una predicción usando el modelo cargado.
    Args:
        features_list (list): Lista de 30 características numéricas.
    Returns:
        dict: Diccionario con 'prediction', 'confidence' y 'label'.
    Raises:
        ValueError: Si la entrada no es válida.
    """
    # Validar longitud
    if len(features_list) != 30:
        raise ValueError(f"Se requieren exactamente 30 características. Recibidas: {len(features_list)}")

    # Convertir a float (valida que sean números)
    try:
        features = [float(x) for x in features_list]
    except (TypeError, ValueError):
        raise ValueError("Todas las características deben ser números")

    # Convertir a array de NumPy
    features_array = np.array(features).reshape(1, -1)

    # Predicción
    pred = modelo.predict(features_array)
    prob = modelo.predict_proba(features_array).max()

    result = {
        "prediction": int(pred[0]),
        "confidence": round(float(prob), 4),
        "label": "benigno" if pred[0] == 1 else "maligno"
    }

    return result
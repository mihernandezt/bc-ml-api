# app/main.py
"""
API Flask para predicción de cáncer de mama usando un modelo RandomForest previamente entrenado.

Endpoints:
- GET /: Health check
- POST /predict: Recibe un JSON con una lista de 30 características numéricas y devuelve la predicción.

Ejemplo de solicitud POST a /predict:
{
  "features": [17.99, 10.38, 122.8, ...]  # 30 valores
}

Respuesta exitosa:
{
  "prediction": 1,
  "confidence": 0.96,
  "label": "benigno"
}
"""

import logging
from flask import Flask, request, jsonify
from .predict import make_prediction

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def health_check():
    """Endpoint de salud: devuelve estado OK si la API está operativa."""
    return jsonify({
        "status": "ok",
        "message": "API de predicción de cáncer de mama activa"
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de predicción.
    Espera un JSON con clave 'features' que contenga una lista de 30 números (float/int).
    Devuelve la clase predicha, la confianza y la etiqueta legible.
    """
    try:
        data = request.get_json()

        # Validar que el cuerpo no esté vacío
        if not data:
            return jsonify({"error": "Cuerpo de la solicitud vacío"}), 400

        if 'features' not in data:
            return jsonify({"error": "Falta el campo 'features'"}), 400

        features = data['features']

        # Validar que sea una lista
        if not isinstance(features, list):
            return jsonify({"error": "El campo 'features' debe ser una lista"}), 400

        # Usar la función de predicción modular
        result = make_prediction(features)

        logger.info(f"Predicción exitosa: {result}")
        return jsonify(result), 200

    except ValueError as ve:
        # Errores de validación esperados
        logger.warning(f"Error de validación en /predict: {ve}")
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        logger.exception("Error no manejado en /predict")
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    # Este bloque solo se ejecuta si se llama directamente a main.py (poco común en producción)
    app.run(host='0.0.0.0', port=5000, debug=False)
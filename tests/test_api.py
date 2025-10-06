# tests/test_api.py
"""
Pruebas automatizadas para la API Flask de predicción de cáncer de mama.

Estas pruebas verifican:
- El endpoint de salud (/)
- La predicción con datos válidos
- El manejo de errores ante entradas inválidas (faltantes, mal formadas, etc.)

Se usa el cliente de prueba integrado de Flask para simular solicitudes HTTP
sin necesidad de levantar un servidor real.
"""

import sys
import os
import pytest

# Añadir la carpeta raíz del proyecto al PATH para poder importar 'app'
# Esto permite ejecutar las pruebas desde cualquier directorio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar la aplicación Flask desde app/main.py
from app.main import app

# Datos de ejemplo válidos: una muestra real del dataset Breast Cancer (clase benigna)
# Contiene exactamente 30 características numéricas, como espera el modelo
VALID_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,
    0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
    0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]


@pytest.fixture
def client():
    """
    Fixture de pytest que proporciona un cliente de prueba de Flask.
    Este cliente simula peticiones HTTP (GET, POST, etc.) sin necesidad
    de iniciar un servidor real. Es ideal para pruebas unitarias e integración.
    """
    app.config['TESTING'] = True  # Activa modo de pruebas (mejor manejo de errores)
    return app.test_client()


# --- PRUEBAS INDIVIDUALES ---

def test_health_check(client):
    """Prueba el endpoint GET / (health check). Debe devolver estado 200 y un JSON con 'status: ok'."""
    response = client.get('/')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'ok'


def test_predict_valid_input(client):
    """Prueba una predicción con entrada válida. Debe devolver 200 y un resultado con predicción, confianza y etiqueta."""
    response = client.post('/predict', json={'features': VALID_FEATURES})
    assert response.status_code == 200
    json_data = response.get_json()
    # Verificar que la respuesta contiene los campos esperados
    assert 'prediction' in json_data
    assert 'confidence' in json_data
    assert 'label' in json_data
    # Verificar que los valores están en rangos válidos
    assert json_data['prediction'] in [0, 1]  # 0 = maligno, 1 = benigno
    assert 0 <= json_data['confidence'] <= 1  # La confianza debe estar entre 0 y 1


def test_predict_missing_features_key(client):
    """Prueba una solicitud sin el campo 'features'. Debe devolver error 400."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_features_not_a_list(client):
    """Prueba cuando 'features' no es una lista (ej. string). Debe devolver error 400."""
    response = client.post('/predict', json={'features': "esto no es una lista"})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_invalid_length(client):
    """Prueba con una lista que no tiene 30 características. Debe devolver error 400."""
    response = client.post('/predict', json={'features': [1.0, 2.0]})  # Solo 2 valores
    assert response.status_code == 400
    error_msg = response.get_json()['error']
    assert '30' in error_msg  # El mensaje debe mencionar que se necesitan 30


def test_predict_non_numeric_values(client):
    """Prueba con valores no numéricos en 'features'. Debe devolver error 400."""
    bad_features = VALID_FEATURES.copy()
    bad_features[0] = "texto_no_valido"  # Inyectar un valor inválido
    response = client.post('/predict', json={'features': bad_features})
    assert response.status_code == 400
    assert 'error' in response.get_json()
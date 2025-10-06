# tests/test_api.py
import unittest
from app import app  # Asegúrate de que tu aplicación Flask se llame 'app' y esté en app.py

class TestAPI(unittest.TestCase):
    def setUp(self):
        """Configura el cliente de prueba antes de cada test."""
        self.client = app.test_client()
        self.client.testing = True

    def test_health_check(self):
        """Verifica que el endpoint raíz devuelva un estado saludable."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("status", json_data)
        self.assertEqual(json_data["status"], "healthy")

    def test_predict_valid(self):
        """Prueba una predicción válida con 30 features numéricas."""
        payload = {"features": [1.0] * 30}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("label", data)
        # Opcional: verificar que prediction sea un número
        self.assertIsInstance(data["prediction"], (int, float))

    def test_predict_missing_features_key(self):
        """Prueba que falle si no se envía la clave 'features'."""
        payload = {}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_wrong_number_of_features(self):
        """Prueba que falle si el número de features no es 30."""
        payload = {"features": [1, 2]}  # Solo 2 features
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_non_numeric_features(self):
        """Prueba que falle si alguna feature no es numérica."""
        payload = {"features": ["a"] * 30}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_with_null_values(self):
        """Prueba que falle si hay valores nulos en features."""
        payload = {"features": [None] * 30}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == '__main__':
    unittest.main()
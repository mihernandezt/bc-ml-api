# fintech-ml-api

API REST de Machine Learning para la predicción diagnóstica de cáncer de mama (benigno/maligno), utilizando el dataset Breast Cancer Wisconsin (Diagnostic).Es un ejemplo sin validez clínica.
Este proyecto implementa un flujo MLOps básico pero completo: entrenamiento del modelo, exposición como API con Flask, validación de entradas, manejo de errores, pruebas automatizadas con pytest, contenerización con Docker y pipeline de CI/CD con GitHub Actions.

## Información del modelo

- **Dataset**: [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
  - Muestras totales: 569  
  - Clases:  
    - `0` → maligno (212 muestras)  
    - `1` → benigno (357 muestras)  
  - Características: 30 (todas reales y positivas)  
  - Tipo: Clasificación binaria

- **Modelo**: `RandomForestClassifier` (scikit-learn)  
- **Métricas típicas** (división 80/20 train/test):  
  - Accuracy: ~97%  
  - Precisión (benigno): ~98%  
  - Recall (maligno): ~95%

- **Características utilizadas**: Las 30 originales del dataset, en el orden exacto definido por `sklearn.datasets.load_breast_cancer().feature_names`.

El modelo se entrena una sola vez y se serializa en `model/model.pkl`.  
No se requiere archivo CSV: los datos se cargan directamente desde `sklearn`.

## Estructura del proyecto

```bash
fintech-ml-api/
├── data/
│   └── breast_cancer.csv          # Opcional: solo si se desea guardar localmente
├── model/
│   ├── train_model.py             # Entrena y guarda el modelo
│   └── model.pkl                  # Modelo serializado
├── app/
│   ├── __init__.py
│   ├── main.py                    # API Flask
│   └── predict.py                 # Lógica de predicción
├── tests/
│   └── test_api.py                # Pruebas automatizadas
├── run.py                         # Punto de entrada para desarrollo local
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml      # CI/CD con GitHub Actions
├── README.md
└── reporte.pdf                   # Documento técnico detallado
```

## Requisitos

- Python 3.10 o superior  
- pip  
- Docker

## Instalación local

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/fintech-ml-api.git
```

2. Crea y activa un entorno virtual (recomendado):

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. (Opcional) Entrena el modelo:

```bash
python model/train_model.py
```

Esto generará `model/model.pkl`. Si ya existe, puedes omitir este paso.

5. Ejecuta la API:

```bash
python run.py
```

La API estará disponible en:  
http://localhost:5000

## Ejecutar pruebas

```bash
pytest tests/
```

## Uso con Docker

1. Construye la imagen:

```bash
docker build -t fintech-ml-api .
```

2. Ejecuta el contenedor:

```bash
docker run -p 5000:5000 fintech-ml-api
```

3. Accede a la API en:  
http://localhost:5000

## Endpoints

### GET /

Devuelve un mensaje de bienvenida.

```json
{
  "status": "ok",
  "message": "API de predicción de cáncer de mama activa"
}
```

### POST /predict

Recibe un JSON con la clave `"features"` que contiene un arreglo de 30 números (en el orden exacto del dataset de sklearn).

```json
{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
```

Respuesta exitosa:

```json
{
  "prediction": 1,
  "confidence": 0.9623,
  "label": "benigno"
}
```
## Manejo de errores

La API valida en el endpoint /predict que:
- El cuerpo de la solicitud sea un JSON válido y no esté vacío.
- Exista la clave "features".
- "features" sea una lista.
- La lista tenga exactamente 30 elementos.
- Todos los elementos sean números (enteros o flotantes).

Ejemplo de error:
```
{
  "error": "Cuerpo de la solicitud vacío"
}
```
## Cómo hacer una predicción manual

Con la API en ejecución (`python run.py` o con Docker), puedes usar `curl`.

**En Windows (PowerShell):**

```bash
curl -Uri "http://localhost:5000/predict" -Method Post -ContentType "application/json" -Body '{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}'
```

En CMD, asegúrate de que el JSON esté en una sola línea.

**En Linux / macOS (Terminal):**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      17.99, 10.38, 122.8, 1001.0, 0.1184,
      0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
      1.095, 0.9053, 8.589, 153.4, 0.006399,
      0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
      25.38, 17.33, 184.6, 2019.0, 0.1622,
      0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
  }'
```

Respuesta esperada:

```json
{
  "prediction": 1,
  "label": "benigno"
}
```

## Entrega

El archivo `reporte.pdf` incluye una explicación detallada del enfoque, metodología, resultados, métricas (precisión, recall, F1) y decisiones técnicas.

## Autor

- [mihernandezt](https://github.com/mihernandezt)

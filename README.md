# fintech-ml-api

API simple de Machine Learning para predicción de diagnóstico de cáncer de mama (benigno/maligno) usando el dataset de Wisconsin.  
Este proyecto muestra un flujo básico pero completo: entrenamiento de un modelo, exposición como API REST con Flask, pruebas automatizadas, contenerización con Docker y un pipeline de CI/CD mínimo.

---

## Informacion del modelo

- Dataset: [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
  - Muestras totales: 569  
  - Clases: 2 (malignant = 212, benign = 357)  
  - Características: 30 (valores reales y positivos)  
  - Tipo: Clasificación binaria  
- Etiquetas:  
  - 0 → maligno  
  - 1 → benigno
- Modelo: Random Forest Classifier (entrenado con sklearn).
- Exactitud (accuracy): ~97% (puede variar ligeramente por división train/test).
- Características usadas: Las 30 del dataset original (no se realizó selección de características).

> El modelo está serializado en model/model.pkl y se carga en la API al iniciar.  
> No se requiere archivo CSV: el dataset se carga directamente desde sklearn.datasets.load_breast_cancer().

---

## Estructura del proyecto
```
fintech-ml-api/
├── data/
│   └── breast_cancer.csv          # No es necesario (solo si se desea guardar localmente)
├── model/
│   ├── train_model.py             # Entrena y guarda el modelo
│   └── model.pkl                  # Modelo serializado
├── app/
│   ├── __init__.py
│   ├── main.py                    # API Flask
│   └── predict.py                 # Lógica de predicción (opcional, modular)
├── tests/
│   └── test_api.py                # Pruebas básicas
├── run.py                         # Punto de entrada para desarrollo local
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml       # CI/CD con GitHub Actions
├── README.md
└── reporte.pdf                    # Documento de entrega con explicación
```
---

## Requisitos

- Python 3.10+
- pip
- Docker

---

## Instalación local

1. Clona el repositorio:
   git clone https://github.com/tu-usuario/fintech-ml-api.git
   cd fintech-ml-api

2. Crea y activa un entorno virtual (recomendado):
   - Windows (CMD/PowerShell):
    ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - Linux / macOS:
    ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Instala las dependencias:  
   `pip install -r requirements.txt`

4. (Opcional) Entrena el modelo:
   python model/train_model.py
   > Esto generará model/model.pkl. Si ya existe, puedes omitir este paso.

5. Ejecuta la API:
   python run.py

6. La API estará disponible en:  
   http://localhost:5000

---

## Ejecutar pruebas

pytest tests/

---

## Usar con Docker

1. Construye la imagen:
   docker build -t fintech-ml-api .

2. Ejecuta el contenedor:
   docker run -p 5000:5000 fintech-ml-api

3. Accede a la API en:  
   http://localhost:5000

---

## Endpoints

GET /
Devuelve un mensaje de bienvenida.

POST /predict
Recibe un JSON con la clave "features" que contiene un array de 30 números (en el orden exacto del dataset de sklearn).

> El modelo espera exactamente 30 características en este orden.

---

## Cómo hacer una predicción manual

Con la API corriendo (python run.py o contenedor Docker), usa uno de los siguientes comandos.

En Windows (PowerShell):
```
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
> Si usas CMD, asegúrate de que el JSON esté en una sola línea sin saltos.

En Linux / macOS (Terminal):
```
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
```
{"prediction": 1, "label": "benigno"}
```
---

## Entrega

El archivo reporte.pdf contiene la explicación detallada del enfoque, metodología, resultados, métricas completas (precisión, recall, F1) y decisiones técnicas.

---

## Autor

- [mihernandezt](https://github.com/mihernandezt)
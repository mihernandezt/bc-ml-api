"""
Entrena un modelo de clasificación para el dataset de cáncer de mama (breast cancer).
Se utiliza RandomForest por su buen equilibrio entre rendimiento, interpretabilidad
y robustez ante características redundantes o ruidosas.

Decisiones clave:
- Se usa GridSearchCV para optimizar hiperparámetros (n_estimators, max_depth).
- Se evalúa con accuracy, precisión, recall y F1-score para evitar sesgo por clases desbalanceadas.
- Se fija random_state para reproducibilidad.
- El modelo se guarda con joblib para uso posterior en la API Flask.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# Aseguramos que el directorio 'model' exista
os.makedirs("model", exist_ok=True)

# Cargar datos
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify mantiene proporción de clases
)

# Definir modelo base
rf = RandomForestClassifier(random_state=42)

# Definir espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Búsqueda de hiperparámetros con validación cruzada (5-fold)
print("Buscando mejores hiperparámetros...")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',  # F1 es robusto para datasets con posible desbalance (aunque este está balanceado)
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
modelo = grid_search.best_estimator_
print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# Evaluar en conjunto de prueba
y_pred = modelo.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

print("\nMétricas en conjunto de prueba:")
for name, value in metrics.items():
    print(f"{name.capitalize()}: {value:.4f}")

print("\nReporte detallado:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Guardar modelo
joblib.dump(modelo, 'model/model.pkl')
print("\nModelo guardado en model/model.pkl")
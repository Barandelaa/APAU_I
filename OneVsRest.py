# Imports necesarios
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Cargar los datos
# Ajustamos el separador de columnas y eliminamos el header
data = pd.read_csv('yeast.data', sep='\s+', header=None)

# Definir atributos (X) y etiquetas (y)
X = data.iloc[:, 1:9]  # Omitimos la primera columna con identificadores (ADT1_YEAST, etc.) y solo añadimos las 8 con atributos
y = data.iloc[:, 9]  # La última columna es la etiqueta

# Convertir los datos en arrays de NumPy
X = np.array(X)
y = np.array(y)

# Normalización de los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Realizamos la primera división: 60% entrenamiento, 40% a dividir después
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=44658877)

# Dividimos el 40% restante en validación (20%) y prueba (20%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=44658877)

# Crear el modelo Logistic Regression con esquema One vs All (OvA)
ova_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))

# Ajustar el modelo con los datos de entrenamiento
ova_clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_ova = ova_clf.predict(X_test)

# Evaluar el modelo
print("Accuracy OvA:", accuracy_score(y_test, y_pred_ova))
print("Confusion Matrix OvA:\n", confusion_matrix(y_test, y_pred_ova))
print("Classification Report OvA:\n", classification_report(y_test, y_pred_ova, zero_division=1))

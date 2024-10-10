#Imports necesarios
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Cargar los datos
# Ajustamos el separador de columnas y eliminamos el header
data = pd.read_csv('yeast.data', sep='\s+', header=None)

X = data.iloc[:, 1:9]  # Omitimos la primera columna con identificadores (ADT1_YEAST, etc.) y solo añadimos las 8 con atributos
# La última columna es la etiqueta (y)
y = data.iloc[:, 9]
# Convertir los datos en arrays de NumPy
X = np.array(X)
y = np.array(y)

# Realizamos la primera división: 60% entrenamiento, 40% a dividir después
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=30)

# Dividimos el 40% restante en validación (20%) y prueba (20%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=30)

# Crear el modelo KNN
knn = KNeighborsClassifier()

# Definir un esquema de cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=30)

# Usar GridSearchCV para encontrar los mejores hiperparámetros (opcional)
param_grid = {'n_neighbors': np.arange(1, 15)}
grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
grid.fit(X_train, y_train)

# Mejor parámetro
print(f"Mejor parámetro k: {grid.best_params_['n_neighbors']}")

# Predecir en el conjunto de prueba
y_pred = grid.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,zero_division=1))

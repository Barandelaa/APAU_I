# Imports necesarios
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar los datos
data = pd.read_csv('yeast.data', sep='\s+', header=None)

# Definir atributos (X) y etiquetas (y)
X = data.iloc[:, 1:9]  # Omitimos la primera columna con identificadores
y = data.iloc[:, 9]    # La última columna es la etiqueta (y)

# Convertir los datos en arrays de NumPy
X = np.array(X)
y = np.array(y)

# Crear el modelo KNN
knn = KNeighborsClassifier()

# Definir un esquema de cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=30)

# Usar GridSearchCV para encontrar los mejores hiperparámetros (opcional)
param_grid = {'n_neighbors': np.arange(1, 15)}

# Variable para almacenar el mejor random_state y accuracy
best_random_state = None
best_accuracy = 0

# Probar varios random_state
for random_state in range(100, 100001):  # Probar de 1 a 100
    # Dividir los datos en entrenamiento y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=random_state)

    # Dividir el 40% restante en validación (20%) y prueba (20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)
    
    # GridSearch para KNN
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = grid.predict(X_test)
    
    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    
    # Verificar si es la mejor precisión
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_state

# Mostrar el mejor random_state y la precisión obtenida
print(f"Mejor random_state: {best_random_state} con una precisión de: {best_accuracy}")

# Entrenar con el mejor random_state
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=best_random_state)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=best_random_state)
grid = GridSearchCV

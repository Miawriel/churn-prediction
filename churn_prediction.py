# Carga de Datos
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.info())

# Exploración de la variable objetivo
print(df['Exited'].value_counts()) 
# Esto te dirá cuántos clientes se fueron (1) vs. se quedaron (0).
# Importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Suponiendo que tu DataFrame se llama 'df'

## 1. ELIMINAR COLUMNAS INNECESARIAS
print("Paso 1: Eliminando identificadores...")
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

## 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS (One-Hot Encoding)
# Las columnas 'Geography' y 'Gender' son de tipo 'object' (texto) y deben ser numéricas.
# Usamos drop_first=True para evitar la trampa de variables dummy (multicolinealidad).
print("Paso 2: Codificando variables categóricas...")
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
print("Columnas después de codificación:", df.columns.tolist())
print("-" * 30)


## 3. SEPARACIÓN DE X (PREDICTORAS) e Y (OBJETIVO)
# X son todas las columnas menos 'Exited'.
X = df.drop('Exited', axis=1)
# y es la columna que queremos predecir.
y = df['Exited']
print("Paso 3: Separando X e y. X ahora tiene", X.shape[1], "columnas.")


## 4. DIVISIÓN DE DATOS (Entrenamiento y Prueba)
# Dividimos los datos antes de escalar para evitar 'Data Leakage'.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Paso 4: Datos divididos. Entrenar:", X_train.shape, "Prueba:", X_test.shape)


## 5. ESCALADO DE VARIABLES NUMÉRICAS
# Escalamos para que variables como 'EstimatedSalary' no dominen sobre 'NumOfProducts'.
scaler = StandardScaler()

# 5a. Entrenamos el scaler solo con los datos de ENTRENAMIENTO
X_train_scaled = scaler.fit_transform(X_train)

# 5b. Aplicamos esa misma transformación a los datos de PRUEBA
X_test_scaled = scaler.transform(X_test)
print("Paso 5: Variables numéricas escaladas.")
print("-" * 30)

print("¡✅ PREPROCESAMIENTO COMPLETO!")

#Entrenando el modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

# 1. Instanciar (crear) el modelo
# Usamos random_state para que los resultados sean reproducibles.
rf_model = RandomForestClassifier(random_state=42)

# 2. Entrenar el modelo con los datos ESCALADOS de entrenamiento
print("Paso 6: Entrenando el modelo Random Forest...")
rf_model.fit(X_train_scaled, y_train)

print("✅ Modelo entrenado con éxito.")

#Prediciendo con el modelo Random Forest
# 3. Realizar predicciones sobre el conjunto de PRUEBA
y_pred_rf = rf_model.predict(X_test_scaled)
print("Predicciones generadas.")


#Rendimiento
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("\n--- Resultados de Evaluación ---")

# a) Matriz de Confusión: Muestra los aciertos y errores
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_rf))
#   [ [Verdaderos Negativos (Correcto: Se queda)], [Falsos Positivos (Error: Dijo que se va, pero se queda)] ]
#   [ [Falsos Negativos (Error: Dijo que se queda, pero se va)], [Verdaderos Positivos (Correcto: Se va)] ]


# b) Reporte de Clasificación (Accuracy, Precision, Recall, F1-Score)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))
# Fíjate especialmente en el 'Recall' para la clase '1' (Exited).

# c) AUC (Area Under the Curve): Mide la capacidad de distinguir las clases
auc_score = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

#Factores de Riesgo
# Obtener la importancia de las características del modelo Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# Mostrar las 10 características más importantes
top_10_features = feature_importances.nlargest(10)
print("\n--- Top 10 Factores de Riesgo (Feature Importance) ---")
print(top_10_features)

# (Opcional: Visualiza los resultados)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh')
plt.title('Importancia de las Características en la Predicción de Churn')
plt.show()

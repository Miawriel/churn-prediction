# 📊 Predicción de Deserción de Clientes Bancarios (Churn)

**Autor/a:** Gabriela Mariel Lopez Armenta

## 🎯 Objetivo del Proyecto

El proyecto busca identificar los principales factores de riesgo que impulsan la deserción (**churn**) de clientes bancarios y desarrollar el modelo de **Machine Learning** más eficiente para predecir a aquellos clientes con alta probabilidad de abandonar el banco.

## 💡 Hallazgos Clave

1. **Modelo Ganador:** El modelo **XGBoost Classifier** fue el seleccionado, siendo el único que superó la métrica crítica del **Recall** $\geq 0.50$ para la clase "Churn" (obteniendo 0.55).

2. **Validación de Hipótesis:** La hipótesis de que el compromiso financiero es la causa es validada, aunque la **Edad del cliente** se reveló como el factor individual más determinante para el **churn**, seguido por el Balance y el Número de Productos.

### Comparación de Métricas Clave

| **Métrica Crítica** | **Regresión Logística** | **Random Forest** | **XGBoost (Ganador)** | 
| :--- | :--- | :--- | :--- | 
| **Recall (Clase 1)** | 0.20 | 0.47 | **0.55** | 
| **AUC Score** | 0.77 | 0.87 | **0.85** | 

## ⚙️ Cómo Ejecutar el Proyecto

Para replicar los resultados y generar los gráficos de Importancia de Características y la Curva ROC:

1. **Clonar el Repositorio:**

   ```bash
   git clone [https://github.com/Miawriel/proyecto-churn-bancario](https://github.com/Miawriel/proyecto-churn-bancario)
   cd proyecto-churn-bancario
Instalar Dependencias: Asegúrate de tener Python instalado.

Bash

pip install -r requirements.txt


Archivos de Datos: Coloca el archivo Churn_Modelling.csv en la carpeta raíz.


Ejecutar el Script: El script completo generará el entrenamiento de 3 modelos y guardará los gráficos.

Bash

python proyecto_churn_final.py





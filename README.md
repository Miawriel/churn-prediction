📊  Predicción de Deserción de Clientes Bancarios (Churn)
Autor/a: Gabriela Mariel Lopez Armenta🎯

Objetivo del Proyecto
El proyecto busca identificar los principales factores de riesgo que impulsan la deserción (\textit{churn}) de clientes bancarios y desarrollar el modelo de \textit{Machine Learning} más eficiente para predecir a aquellos clientes con alta probabilidad de abandonar el banco.

💡 Hallazgos Clave
Modelo Ganador: El modelo XGBoost Classifier fue el seleccionado.
Criterio de Éxito: El modelo fue el \textbf{único} en superar la métrica crítica del Recall $\geq 0.50$ para la clase "Churn".
Factores de Riesgo (Feature Importance): La \textbf{Edad del cliente} resultó ser el factor individual más determinante para el \textit{churn}, seguido por el Balance y el Número de Productos.
| Métrica Crítica | Regresión Logística | Random Forest | XGBoost (Ganador) || Recall (Clase 1) | 0.20 | 0.47 | 0.55 || AUC Score | 0.77 | 0.87 | 0.85 |

⚙️ Cómo Ejecutar el Proyecto
Para replicar los resultados y generar los gráficos de Importancia de Características y la Curva ROC:
Clonar el Repositorio:git clone [https://github.com/Miawriel/proyecto-churn-bancario](https://github.com/Miawriel/proyecto-churn-bancario)
cd proyecto-churn-bancario
Instalar Dependencias:
Asegúrate de tener Python instalado.pip install -r requirements.txt
Archivos de Datos:Coloca el archivo Churn_Modelling.csv en la carpeta raíz.
Ejecutar el Script:El script completo generará el entrenamiento de 3 modelos y guardará los gráficos.
python proyecto_churn_final.py

📂 Estructura del Repositorio
README.md: Este archivo.
proyecto_churn_final.py: Código fuente con el preprocesamiento, entrenamiento y visualizaciones.
requirements.txt: Dependencias de Python.
Propuesta_Proyecto.pdf: Documento PDF de la propuesta inicial (LaTeX).
Reporte_Final_Bancario.pdf: Reporte final detallado con conclusiones de negocio (LaTeX).




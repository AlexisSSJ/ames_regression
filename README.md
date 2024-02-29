### **Modelo predictivo de precios de casas en Ames, Iowa**

**Descripción:**

Este proyecto presenta un modelo predictivo desarrollado en Python para estimar el precio de venta de viviendas en Ames, Iowa. Se utilizan técnicas de Machine Learning para analizar un conjunto de datos con información sobre las propiedades (superficie, habitaciones, ubicación, etc.).

**Contenido:**

-   **data:** Contiene los conjuntos de datos utilizados:

    -   `proyecto_ames_train.xlsx`: Datos etiquetados para entrenamiento.

    -   `proyecto_ames_test.csv`: Datos no etiquetados para evaluación.

    -   `predictions.csv`: Predicciones del modelo final.

-   **model:** Contiene carpetas con archivos del análisis de modelos no paramétricos (KNN y Random Forest):

    -   `knn`: contiene resultados de importancia de variables y cross-validation para la selección de hiper-parámetros para el modelo de KNN.

    -   `Random_forest`: contiene resultados de importancia de variables y cross-validation para la selección de hiper-parámetros para el modelo de RF.

-   **src:** Contiene los scripts con el desarrollo del proyecto:

    -   Particionamiento de datos.

        -   `01_Part_data.py`

    -   Funciones creadas para el análisis.

        -   `02_custom_functions.py`

    -   Preprocesamiento de datos.

        -   `03_data_cleaning.py`, `04_data_preprocesado.py`

    -   Ajuste de diferentes modelos.

        -   `05_linear_regression.py`, `06_knn_regression.py`, `07_random_forest_regression.py`,

    -   Predicciones finales

        -   `08_test_predictions\.py`

    -   Reporte del proyecto.

        -   `EDA.qmd`

**Herramientas:**

-   Manejo de datos : Pandas y Siuba

-   Visualización: Plotnine

-   Analisis estadistico y Machine Learning: Scikit-learn y Statsmodels

**Resultados:**

-   Para las predicciones finales se utilizó la regresión líneal, ya que era un modelo bastante simple y con buenos resultados, el valor que arrojó como resultado final fue de $0.83$.

-   $R^2_{adj} = 0.91$ en test, $0.90$ en validación.

-   Identificación de los factores que más influyen en el precio (Variables de superficies en pies cuadrados, proporcines relativas al tamaño del lote, cantidad de baños, recamaras, chimeneas, etc.).

**Próximos pasos:**

-   Mejorar la estructura del reporte.

-   Intentar stacking.

-   Creación de presentación tipo profesional resumiendo resultados.

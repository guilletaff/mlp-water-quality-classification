# Clasificación de Calidad y Potabilidad del Agua con Redes Neuronales (MLP)

Este repositorio contiene el proyecto final para la asignatura **Redes Neuronales y Aprendizaje Profundo** en **CUNEF**. El objetivo principal es clasificar y predecir si una muestra de agua es potable (1) o no (0) a partir de sus parámetros físico-químicos, empleando un modelo de **Perceptrón Multicapa (MLP)** optimizado.

---

## Autores
* **Guillermo Taffouraud González-Palacios**
* **Adriana Pérez Cembranos**

---

## Estructura del Proyecto

El proyecto está compuesto por los siguientes archivos dentro de la carpeta `RN_WaterPotability`:

*  **`water_potability.csv`**: Dataset original extraído de Kaggle. Contiene 3277 registros y 10 columnas (9 variables predictoras físico-químicas y 1 variable objetivo).
*  **`water_preprocessed.csv`**: Conjunto de datos final preprocesado, imputado y filtrado tras aplicar la selección de las 7 variables más influyentes.
*  **`MLP_WaterQuality.ipynb`**: Jupyter Notebook con todo el flujo de trabajo en Python:
  * Eliminación de duplicados e imputación avanzada.
  * Selección de atributos usando Información Mutua.
  * Análisis Exploratorio de Datos (EDA) con histogramas, matrices de correlación, scatter plots y boxplots.
  * División train-test y balanceo por sobremuestreo (upsampling).
  * Entrenamiento del Pipeline MLP con ajuste de hiperparámetros.
  * Validación cruzada estratificada (10-Fold CV).
  * Evaluación mediante matriz de confusión, reporte de clasificación y curvas ROC.
*  **`Redes Neuronales. Water Quality and Potability .pdf`**: Memoria técnica detallada del proyecto (16 páginas) que explica la justificación teórica, decisiones de diseño, evolución experimental, análisis de resultados y conclusiones.

---

##  Metodología y Preprocesamiento

Dado que el dataset presenta características químicas complejas y valores ausentes en variables clave (`pH`, `Sulfate` y `Trihalomethanes`), se aplicaron los siguientes tratamientos:

1. **Eliminación de Duplicados**: Ejecutado mediante `df.drop_duplicates()`.
2. **Imputación mediante Vecinos Cercanos**: Se implementó **`KNNImputer` (K=5, weights="distance")** para estimar los valores nulos basándose en la distancia a las muestras más similares, evitando la pérdida de varianza que ocasionaría una imputación por media o mediana.
3. **Normalización/Escalado**: Se aplicó **`MinMaxScaler`** para proyectar todas las variables a un rango $[0, 1]$. Esto evita que variables con escalas y magnitudes elevadas (como `Solids`) monopolicen el cálculo de gradientes y distorsionen los pesos de la red.
4. **Selección de Atributos**: Se evaluaron ANOVA F-value (lineal) e **Información Mutua (Mutual Information, no lineal)**. Dado el bajo grado de correlación lineal y la naturaleza no lineal del MLP, se optó por la Información Mutua, seleccionando los **7 atributos más importantes**:
   * `Hardness`, `Sulfate`, `Conductivity`, `ph`, `Organic_carbon`, `Turbidity` y `Solids`.
   * *(Las variables `Chloramines` y `Trihalomethanes` se excluyeron al no aportar suficiente valor predictivo).*

---

##  Arquitectura de la Red Neuronal y Entrenamiento

El modelo óptimo se implementó utilizando un Pipeline de `scikit-learn` que incluye imputación mediana secundaria, escalado estándar y un clasificador de red neuronal de tipo **Multi-Layer Perceptron (MLP)**:

```python
modelo = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), # Estructura de pirámide invertida
        activation="relu",
        solver="adam",
        alpha=1e-5,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42
    ))
])
```

### Decisiones de Diseño Clave:
* **Capas Ocultas**: Arquitectura en **pirámide invertida (128 ➔ 64 ➔ 32 neuronas)**. La primera capa expande la dimensionalidad y extrae patrones complejos de forma amplia; las capas sucesivas reducen y refinan la abstracción de rasgos químicos.
* **Activación ReLU**: Aporta no linealidad y eficiencia de cálculo, mitigando el desvanecimiento de gradientes.
* **Optimizador Adam**: Ajusta de forma adaptativa la tasa de aprendizaje para cada peso individual a partir de los momentos del gradiente.
* **Early Stopping**: Detiene el proceso si no hay mejora en el 10% de validación durante 30 iteraciones consecutivas, evitando el sobreajuste (*overfitting*).
* **Balanceo de Clases**: Para compensar el desbalance de muestras potables (1) y no potables (0), se aplicó **upsampling** de la clase 1 en el conjunto de entrenamiento usando `resample`.
* **Validación Cruzada Estratificada (10-Fold CV)**: Reemplaza la división simple train-test (80/20) para suprimir el sesgo de partición y garantizar la estabilidad y reproducibilidad de las métricas.

---

##  Evolución de Experimentos

Se realizaron múltiples pruebas con diferentes arquitecturas y técnicas de imputación antes de llegar a la combinación óptima:

| Experimento | Descripción del Experimento / Técnicas | Accuracy Promedio | Observaciones |
| :---: | :--- | :---: | :--- |
| **Exp. 1** | Red básica (16, 8) + Imputación por media. | 62.15% | **Underfitting**: El modelo era demasiado simple para capturar relaciones complejas. |
| **Exp. 2** | Red profunda (256x3) sin regularización. | 68.40% | **Overfitting**: El modelo memorizó el ruido debido a la profundidad sin control. |
| **Exp. 3** | Arquitectura (100, 50) sin MinMaxScaler. | 51.10% | **Inestabilidad**: Las variables con rangos altos (Solids) desestabilizaron el aprendizaje. |
| **Exp. 4** | Configuración (64, 32, 16) + KNNImputer. | 71.87% | **Mejora**: Confirmó que una imputación de nulos más robusta es clave para el rendimiento. |
| **Final** | **Pirámide (128, 64, 32) + K-Fold + Adam (Óptimo)** | **78.91%** | **Modelo Óptimo**: Equilibrio excelente entre capacidad de generalización y estabilidad. |

---

##  Resultados Obtenidos

El modelo final validado mediante **10-Fold Cross-Validation** arrojó métricas altamente satisfactorias:

*  **Accuracy Promedio (10-Fold CV)**: **78.91%**
*  **ROC AUC (Área Bajo la Curva)**: **0.8365** (83.6% de probabilidad de discriminar correctamente entre muestras potables y no potables).
*  **Precision de Clase 0 (No Potable)**: **81%** (Alta confiabilidad cuando el sistema emite una alerta de contaminación/no potabilidad).
*  **Recall de Clase 1 (Potable)**: **82%** (Permite identificar correctamente la gran mayoría de fuentes aptas para el consumo, reduciendo falsos negativos).
*  **Convergencia**: El modelo convergió de manera óptima y estable en **99 iteraciones**.

---

##  Requisitos de Instalación y Ejecución

Para poder correr el código contenido en el notebook `MLP_WaterQuality.ipynb`, es necesario contar con un entorno de Python 3.8+ y las siguientes librerías instaladas:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly
```

Una vez cubiertas las dependencias, basta con abrir el archivo `MLP_WaterQuality.ipynb` en tu IDE preferido (VS Code, Jupyter Lab, etc.) y ejecutar todas sus celdas de forma secuencial.

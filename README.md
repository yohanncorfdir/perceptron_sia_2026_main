# Proyecto SIA 2026 — Perceptrón y Redes Neuronales

---

## Estructura del proyecto

```
perceptron_sia_2026_main/
├── data/
│   ├── digits.csv                  # 12 450 imágenes de entrenamiento (28×28)
│   ├── more_digits.csv             # 15 742 imágenes adicionales
│   ├── digits_test.csv             # 2 498 imágenes de test
│   ├── fraud_dataset.csv           # 7 500 transacciones financieras
│   └── digit_dataset_loader.py     # Utilidades de carga y visualización
├── algorithmEjercicio1/
│   ├── EDA/
│   │   └── eda_fraud.py            # Análisis exploratorio del dataset de fraude
│   ├── perceptronSimpleLineal.py   # Perceptrón lineal (función escalón)
│   ├── perceptronSimpleNoLineal.py # Perceptrón no lineal (sigmoide)
│   ├── perceptron_and.py           # Demostración: puerta lógica AND
│   ├── perceptron_or.py            # Demostración: puerta lógica OR
│   ├── perceptron_fraud.py         # Entrenamiento sobre el dataset de fraude
│   └── generalizacion_fraud.py     # Cross-validation + optimización de umbral
├── algorithmEjercicio2/
│   ├── mlp.py                      # Red neuronal multicapa (MLP v1)
│   ├── digits_mlp.py               # Comparativa de arquitecturas y optimizadores
│   └── show_digit.py               # Visualización de dígitos desde CSV
└── algorithmEjercicio3/
    ├── mlp_v2.py                   # MLP mejorado (ReLU, decay, early stopping, persistencia)
    ├── mejor_digits_mlp.py         # Pipeline completo hacia 98% de exactitud
    └── modelos/                    # Pesos guardados (.npz) — se crean al entrenar
```

---

## Datasets

| Dataset | Muestras | Características | Tarea |
|---|---|---|---|
| `digits.csv` | 12 450 | imagen 784 px + etiqueta 0-9 | Entrenamiento dígitos (Ej. 2-3) |
| `more_digits.csv` | 15 742 | imagen 784 px + etiqueta 0-9 | Entrenamiento adicional (Ej. 3) |
| `digits_test.csv` | 2 498 | imagen 784 px + etiqueta 0-9 | Evaluación final dígitos |
| `fraud_dataset.csv` | 7 500 | 20 variables financieras + `flagged_fraud` | Detección de fraude (Ej. 1) |

**Formato imágenes**: cada imagen se almacena como cadena de texto y se parsea con `ast.literal_eval()`. Los píxeles están normalizados en [0, 1].

**Formato fraude**: variables financieras normalizadas en [0, 1], variable objetivo binaria (0 = legítima, 1 = fraude). Se elimina la columna `big_model_fraud_probability` para evitar fuga de información.

---

## Ejercicio 1 — Perceptrón Simple: Detección de Fraude

### Objetivo

Implementar y comparar dos versiones de perceptrón simple para clasificar transacciones financieras como fraudulentas o legítimas. El ejercicio cubre desde puertas lógicas básicas hasta un problema real desbalanceado.

### Modelos implementados

#### `PerceptronLineal` (`perceptronSimpleLineal.py`)

Perceptrón clásico con función de activación escalón. Actualiza los pesos únicamente cuando se produce un error de clasificación.

```
Regla de actualización: w ← w + alpha * (y - ŷ) * x
```

| Parámetro | Descripción |
|---|---|
| `N` | Dimensión de la entrada |
| `alpha` | Tasa de aprendizaje (por defecto 0.1) |

**Métodos principales**:
- `step(x)` — Función escalón: devuelve 0 o 1 según el signo de la activación
- `fit(X, y, epochs, X_val, y_val)` — Entrenamiento con regla del perceptrón
- `predict(X)` — Predicción binaria

**Limitación**: Solo puede separar clases linealmente separables. No converge en problemas no lineales.

#### `PerceptronNoLineal` (`perceptronSimpleNoLineal.py`)

Perceptrón con activación sigmoide y descenso del gradiente. Permite aprender fronteras de decisión más suaves.

```
Activación:    σ(z) = 1 / (1 + e^(-z))
Actualización: w ← w - alpha * ∂L/∂w    (descenso del gradiente)
```

| Parámetro | Descripción |
|---|---|
| `N` | Dimensión de la entrada |
| `alpha` | Tasa de aprendizaje (por defecto 0.1) |

**Métodos principales**:
- `sigmoid(x)` / `sigmoid_deriv(x)` — Activación y su derivada
- `fit(X, y, epochs, X_val, y_val)` — Entrenamiento por gradiente
- `predict(X, threshold=0.5)` — Predicción con umbral ajustable
- `predict_proba(X)` — Probabilidades crudas (útil para optimizar el umbral)

### Scripts

#### `perceptron_and.py` / `perceptron_or.py`
Demostración sobre puertas lógicas para verificar el funcionamiento básico de ambos perceptrones antes de aplicarlos a datos reales.

#### `perceptron_fraud.py`
Entrena ambos modelos (lineal y no lineal) sobre el dataset de fraude completo y compara su exactitud final. Permite observar la saturación del modelo lineal frente a datos no linealmente separables.

#### `generalizacion_fraud.py`
Estudio de generalización con dos técnicas:

1. **Stratified K-Fold Cross Validation (K=5)**:
   - Divide el dataset manteniendo la proporción de fraudes (~17%) en cada fold
   - Evita que algún fold quede sin casos positivos por azar
   - Devuelve media y desviación estándar de las métricas

2. **Optimización del umbral de decisión**:
   - Barre umbrales de 0.1 a 0.9 sobre las probabilidades predichas
   - Calcula Precisión, Recall y F1 para cada umbral
   - Genera `plot_umbral.png`: curva de métricas vs umbral para elegir el punto óptimo

#### `EDA/eda_fraud.py`
Análisis exploratorio del dataset:
- `plot_correlacion.png`: mapa de calor de correlación entre variables — identifica las más relevantes para predecir fraude
- `plot_boxplots.png`: diagramas de caja separados por clase para detectar outliers y diferencias de distribución

### Métricas reportadas

- **Exactitud** (Accuracy)
- **Precisión** (Precision): fracción de alertas correctas
- **Recall** (Sensibilidad): fracción de fraudes detectados
- **F1-Score**: media armónica de Precisión y Recall

---

## Ejercicio 2 — Red Neuronal Multicapa: Clasificación de Dígitos

### Objetivo

Implementar una red neuronal multicapa (MLP) desde cero y explorar el impacto de distintas arquitecturas, tasas de aprendizaje y optimizadores sobre la clasificación de imágenes de dígitos manuscritos (28×28 px, 10 clases).

### Modelo implementado

#### `MLP` (`algorithmEjercicio2/mlp.py`)

Red neuronal totalmente conectada con capas ocultas configurables, activación sigmoide en capas ocultas y softmax en la capa de salida.

| Parámetro | Descripción |
|---|---|
| `n_entrada` | 784 (imagen 28×28 aplanada) |
| `capas_ocultas` | Lista con el número de neuronas por capa, ej. `[128, 64]` |
| `n_salida` | 10 (clases 0-9) |
| `alpha` | Tasa de aprendizaje |
| `optimizador` | `'sgd'`, `'momentum'` o `'adam'` |

**Métodos principales**:
- `_forward(X)` — Propagación hacia adelante, guarda activaciones para el retropropagación
- `_backward(activaciones, y_onehot)` — Retropropagación del gradiente y actualización de pesos
- `fit(X, y, epochs, batch_size, X_val, y_val, verbose)` — Entrenamiento por mini-lotes
- `predict(X)` — Devuelve la clase con mayor probabilidad (argmax de softmax)
- `score(X, y)` — Exactitud sobre un conjunto dado

**Inicialización de pesos**:

Inicialización Xavier para prevenir la saturación de la sigmoide al inicio del entrenamiento:
```
W ~ Uniforme[-√(6/(n_in + n_out)),  √(6/(n_in + n_out))]
```

**Función de pérdida**: Entropía cruzada (implícita en la derivada de softmax + cross-entropy).

**Codificación de etiquetas**: One-hot encoding — cada etiqueta entera se convierte en un vector de 10 posiciones con un 1 en la posición correspondiente.

### Script de experimentación

#### `digits_mlp.py`

Compara **18 variantes** en dos fases:

**Fase 1 — Exploración de arquitecturas con SGD** (12 variantes):

| Arquitectura | Tasas de aprendizaje probadas |
|---|---|
| `[32, 16]` | 0.1, 0.05 |
| `[64, 32]` | 0.1, 0.05 |
| `[64]` | 0.1, 0.05 |
| `[128]` | 0.1, 0.05 |
| `[128, 64]` | 0.1, 0.05 |
| `[256, 128]` | 0.1, 0.05 |

**Fase 2 — Comparativa de optimizadores** (6 variantes):

| Configuración | Optimizador | Alpha |
|---|---|---|
| `[128, 64]` | SGD | 0.01 |
| `[128, 64]` | Momentum | 0.01 |
| `[128, 64]` | Adam | 0.001 |
| `[256, 128]` | SGD | 0.01 |
| `[256, 128]` | Momentum | 0.01 |
| `[256, 128]` | Adam | 0.001 |

**Salidas generadas**:
- `plot_curvas.png`: curvas de aprendizaje (exactitud en test por epoch) para todas las variantes
- `plot_confusion.png`: matriz de confusión 10×10 del mejor modelo
- Reporte por consola: exactitud, Precisión, Recall y F1 por clase y macro-promedio

### Métricas reportadas

- Exactitud global
- Precisión, Recall y F1 por clase (0-9)
- Macro-F1 (promedio no ponderado entre clases)
- Matriz de confusión

---

## Ejercicio 3 — MLP Avanzado: Objetivo 98% de Exactitud

### Objetivo

Superar el 98% de exactitud en clasificación de dígitos aplicando técnicas avanzadas: activación ReLU, data augmentation, decaimiento del learning rate, early stopping, cross-validation estratificada y persistencia de modelos.

### Modelo implementado

#### `MLPv2` (`algorithmEjercicio3/mlp_v2.py`)

Versión mejorada del MLP con soporte para ReLU, decaimiento de la tasa de aprendizaje, parada temprana y serialización completa de los pesos.

| Parámetro | Descripción |
|---|---|
| `n_entrada` | 784 |
| `capas_ocultas` | Lista de tamaños, ej. `[512, 256]` |
| `n_salida` | 10 |
| `alpha` | Tasa de aprendizaje (por defecto 0.001) |
| `optimizador` | `'sgd'`, `'momentum'` o `'adam'` |
| `activacion` | `'relu'` o `'sigmoid'` |
| `lr_decay` | Factor de decaimiento del learning rate (0 = desactivado) |

**Inicialización de pesos**:
- **ReLU → He initialization**: `std = √(2 / n_entrada_capa)` — diseñada para evitar la saturación con ReLU
- **Sigmoid → Xavier initialization**: `W ~ Uniforme[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]`

**Métodos principales**:
- `fit(X, y, epochs, batch_size, X_val, y_val, paciencia, min_delta, verbose)` — Entrenamiento con early stopping opcional
- `guardar(ruta)` — Guarda pesos, sesgos, hiperparámetros y estado del optimizador en `.npz`
- `cargar(ruta)` *(classmethod)* — Reconstruye el modelo completo desde un `.npz`
- `predict(X)` / `score(X, y)` — Predicción y evaluación

### Script principal

#### `mejor_digits_mlp.py`

Pipeline completo organizado en tres fases:

**Fase 1 — Comparativa base** (3 variantes, sin augmentation)

| Variante | Capas | Activación | Alpha |
|---|---|---|---|
| Referencia sigmoid | `[256, 128]` | Sigmoid | 0.001 |
| ReLU estándar | `[256, 128]` | ReLU | 0.001 |
| ReLU arquitectura grande | `[512, 256]` | ReLU | 0.001 |

Datos: `digits.csv` + `more_digits.csv` → 28 192 muestras de entrenamiento.

**Fase 2 — Data Augmentation** (2 variantes)

| Variante | Capas | Decay |
|---|---|---|
| ReLU + aug | `[256, 128]` | 0.0 |
| ReLU + aug + decay | `[512, 256]` | 0.005 |

Los modelos entrenados se guardan automáticamente en `modelos/`. En ejecuciones posteriores se cargan directamente sin reentrenar.

**Fase 3 — Stratified K-Fold Cross-Validation** (2 variantes, K=5)

Los índices se estratifican manualmente por clase (10 dígitos) para garantizar que cada fold mantenga la distribución original. El modelo se entrena únicamente con los datos del fold de entrenamiento y se evalúa sobre el fold de test. Se reportan exactitud y Macro-F1 por fold, más la media y desviación estándar entre folds.

### Técnicas avanzadas

#### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

Ventajas frente a sigmoide:
- No satura para valores positivos grandes → gradientes más limpios en capas profundas
- Convergencia más rápida en la práctica
- Derivada: 1 si z > 0, 0 si z ≤ 0 (binaria, sin cálculo exponencial)

#### Data Augmentation

Técnica para multiplicar artificialmente el tamaño del dataset desplazando cada imagen 1 píxel en las 4 direcciones cardinales (arriba, abajo, izquierda, derecha), usando `np.roll` sobre la rejilla 28×28.

```
Dataset original: 28 192 muestras
Dataset aumentado: 28 192 × 5 = 140 960 muestras
```

Beneficio: el modelo aprende a reconocer dígitos ligeramente desplazados, mejorando la robustez ante variaciones de posición.

#### Decaimiento del Learning Rate (lr_decay)

```
alpha(epoch) = alpha_ini / (1 + lr_decay × epoch)
```

Reduce progresivamente la tasa de aprendizaje a lo largo del entrenamiento. Los primeros epochs hacen pasos grandes para explorar el espacio de parámetros; los últimos hacen pasos pequeños para afinar la convergencia y evitar oscilaciones.

#### Early Stopping

Para el entrenamiento automáticamente si la exactitud de validación no mejora al menos `min_delta` durante `paciencia` epochs consecutivas.

```python
fit(..., paciencia=10, min_delta=0.001)
```

Beneficios:
- Evita el sobreajuste (overfitting)
- Reduce el tiempo de entrenamiento si el modelo ya ha convergido
- `paciencia=10` con `min_delta=0.001` significa: parar si no hay mejora de 0.1% en 10 epochs seguidas

#### Stratified K-Fold Cross-Validation

Divide el dataset en K bloques (folds) manteniendo la misma proporción de cada clase en cada fold. Para K=5 y 10 clases:

1. Se extraen los índices de cada dígito por separado
2. Se mezclan aleatoriamente dentro de cada clase
3. Se dividen en K bloques iguales por clase
4. Cada fold se construye concatenando un bloque de cada clase

En cada iteración, 1 fold es test y los K-1 restantes son entrenamiento. El proceso se repite K veces para que cada muestra sea evaluada exactamente una vez.

#### Persistencia de modelos (guardar / cargar)

Los modelos se serializan en formato `.npz` (numpy nativo, sin dependencias externas). El archivo contiene:
- Matrices de pesos `W_i` y sesgos `b_i` de cada capa
- Hiperparámetros: arquitectura, `alpha_ini`, `lr_decay`, `activacion`, `optimizador`
- Estado del optimizador: momentos `mW`, `mb`, `vW`, `vb` (Adam) o velocidades `vW`, `vb` (Momentum), contador de pasos `t`

```python
# Guardar después de entrenar
modelo.guardar("modelos/mi_modelo.npz")

# Cargar en la siguiente ejecución
modelo = MLPv2.cargar("modelos/mi_modelo.npz")
```

En `mejor_digits_mlp.py`, el patrón es automático: si el archivo `.npz` existe se carga; si no existe se entrena y se guarda. Para forzar el reentrenamiento basta con eliminar el archivo correspondiente en `modelos/`.

---

## Modelos y optimizadores — Referencia completa

### Funciones de activación

| Función | Fórmula | Rango | Uso |
|---|---|---|---|
| **Escalón** (step) | `1 si z ≥ 0, else 0` | {0, 1} | PerceptronLineal (Ej. 1) |
| **Sigmoide** | `1 / (1 + e^(-z))` | (0, 1) | PerceptronNoLineal + MLP capas ocultas (Ej. 1-2) |
| **ReLU** | `max(0, z)` | [0, +∞) | MLPv2 capas ocultas (Ej. 3) |
| **Softmax** | `e^z_i / Σ e^z_j` | (0, 1), suma 1 | Capa de salida MLP (Ej. 2-3) |

### Optimizadores

#### SGD (Stochastic Gradient Descent)

```
W ← W - alpha × ∂L/∂W
```

Descenso del gradiente puro, sin historial. Simple y predecible, pero puede oscilar y converge lento en superficies de error complejas.

#### Momentum

```
v  ← beta × v_prev + alpha × ∂L/∂W    (beta = 0.9)
W  ← W - v
```

Acumula una "velocidad" exponencialmente ponderada del gradiente. Amortigua las oscilaciones y acelera la convergencia en direcciones consistentes.

#### Adam (Adaptive Moment Estimation)

```
m  ← beta1 × m_prev + (1 - beta1) × g        (beta1 = 0.9)
v  ← beta2 × v_prev + (1 - beta2) × g²        (beta2 = 0.999)
m̂  = m / (1 - beta1^t)     # corrección de sesgo
v̂  = v / (1 - beta2^t)
W  ← W - alpha × m̂ / (√v̂ + eps)              (eps = 1e-8)
```

Combina las ventajas de Momentum (dirección suavizada) y RMSprop (escala adaptativa por peso). En la práctica es el más robusto y el que menos requiere ajuste manual de la tasa de aprendizaje.

### Inicialización de pesos

| Nombre | Fórmula | Activación objetivo |
|---|---|---|
| **Xavier** | `W ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]` | Sigmoide / Tanh |
| **He** | `W ~ N(0, √(2/n_in))` | ReLU |

Una inicialización adecuada previene que las activaciones colapsen a 0 o saturen hacia los extremos desde el primer paso de entrenamiento.

---

## Métricas de evaluación

| Métrica | Fórmula | Cuándo usarla |
|---|---|---|
| **Exactitud** | `aciertos / total` | Dataset balanceado |
| **Precisión** | `TP / (TP + FP)` | Minimizar falsas alarmas |
| **Recall** | `TP / (TP + FN)` | Minimizar fraudes no detectados |
| **F1-Score** | `2 × P × R / (P + R)` | Balance Precisión-Recall |
| **Macro-F1** | `media(F1 por clase)` | Dataset multiclase |

**Matriz de confusión**: tabla N×N que muestra para cada clase real cuántas muestras se predijeron en cada clase. La diagonal principal representa aciertos; fuera de la diagonal, errores.

---

## Ejecución

### Requisitos

El proyecto requiere las siguientes librerías de Python:
- `numpy`
- `pandas`
- `matplotlib`

Para instalarlas, se recomienda configurar un entorno virtual e instalar las dependencias con `pip`:

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python3 -m venv venv

# 2. Activar el entorno virtual
source venv/bin/activate

# 3. Instalar dependencias
pip install numpy pandas matplotlib
```

*(Si prefieres no usar un entorno virtual, puedes instalar las dependencias directamente ejecutando: `pip3 install numpy pandas matplotlib`)*.

Asegúrate de ejecutar los scripts utilizando `python3` si tu sistema operativo (como Linux) por defecto apunta a otra versión de Python al utilizar el comando `python`.

No se utilizan frameworks de deep learning (TensorFlow, PyTorch, scikit-learn). Todo está implementado desde cero con NumPy.

### Ejercicio 1

```bash
# Análisis exploratorio
python algorithmEjercicio1/EDA/eda_fraud.py

# Entrenamiento sobre fraude
python algorithmEjercicio1/perceptron_fraud.py

# Cross-validation y optimización de umbral
python algorithmEjercicio1/generalizacion_fraud.py
```

### Ejercicio 2

```bash
python algorithmEjercicio2/digits_mlp.py
```

### Ejercicio 3

```bash
python algorithmEjercicio3/mejor_digits_mlp.py
```

En la primera ejecución se entrenan todos los modelos y se guardan en `algorithmEjercicio3/modelos/`. Las ejecuciones siguientes cargan los modelos directamente. Para reentrenar un modelo específico, eliminar el archivo `.npz` correspondiente.

### Visualización de dígitos

```bash
# Ver el dígito en la fila 42 del archivo de test
python algorithmEjercicio2/show_digit.py 42 digits_test.csv
```

---

## Resultados esperados

| Ejercicio | Modelo | Dataset | Exactitud aproximada |
|---|---|---|---|
| 1 | PerceptronLineal | Fraude | ~80% |
| 1 | PerceptronNoLineal | Fraude | ~85% |
| 2 | MLP `[256,128]` Adam | Dígitos | ~96-97% |
| 3 | MLPv2 `[512,256]` ReLU + aug + decay | Dígitos | ~98%+ |

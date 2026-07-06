# Laboratorio: Resolución del Perceptrón Multicapa (MLP) para el Problema XOR

Este documento contiene la explicación detallada, el desarrollo matemático paso a paso de una iteración (Forward y Backpropagation) y el código de implementación en Python para resolver la compuerta lógica no lineal **XOR** usando una Red Neuronal Artificial.

---

## 1. Arquitectura de la Red Neuronal

De acuerdo con las especificaciones de la tarea, la red cuenta con la siguiente estructura:

* **Capa de Entrada:** 2 neuronas ($X_1, X_2$).
* **Capa Oculta:** 1 capa con 2 neuronas ($H_1, H_2$).
* **Capa de Salida:** 1 neurona ($Y$).
* **Tasa de Aprendizaje ($\alpha$):** 0.25.
* **Función de Activación:** Función Sigmoide Logística en todas las neuronas:
    $$f(z) = \frac{1}{1 + e^{-z}}$$
    *Su derivada respecto a su entrada es:* $f'(z) = f(z)(1 - f(z))$.



### Datos del Problema (Tabla de Verdad XOR)
Trabajaremos el ejemplo utilizando la tercera fila de la tabla de verdad:
* **Entradas:** $X_1 = 1$, $X_2 = 0$
* **Salida Deseada (Target $T$):** $1$

### Pesos y Biases Iniciales (Valores de ejemplo para la iteración)
* **Pesos de Entrada a Capa Oculta ($W^{(1)}$):**
    * Para $H_1$: $w_{11} = 0.5$, $w_{21} = 0.3$, Bias $b_{h1} = 0.1$
    * Para $H_2$: $w_{12} = 0.2$, $w_{22} = 0.4$, Bias $b_{h2} = 0.2$
* **Pesos de Capa Oculta a Salida ($W^{(2)}$):**
    * Para $Y$: $w_{h1} = 0.7$, $w_{h2} = -0.4$, Bias $b_y = 0.3$

---

## 2. Paso 1: Propagación hacia Adelante (Forward Propagation)

El objetivo es calcular los valores de activación desde la entrada hasta la salida.

### A. Cálculo en la Capa Oculta

**Para la neurona ocultar $H_1$:**
1. Suma neta ($Z_{h1}$):
   $$Z_{h1} = (X_1 \cdot w_{11}) + (X_2 \cdot w_{21}) + b_{h1}$$
   $$Z_{h1} = (1 \cdot 0.5) + (0 \cdot 0.3) + 0.1 = 0.6$$
2. Activación ($A_{h1}$):
   $$A_{h1} = \sigma(0.6) = \frac{1}{1 + e^{-0.6}} \approx 0.6457$$

**Para la neurona oculta $H_2$:**
1. Suma neta ($Z_{h2}$):
   $$Z_{h2} = (X_1 \cdot w_{12}) + (X_2 \cdot w_{22}) + b_{h2}$$
   $$Z_{h2} = (1 \cdot 0.2) + (0 \cdot 0.4) + 0.2 = 0.4$$
2. Activación ($A_{h2}$):
   $$A_{h2} = \sigma(0.4) = \frac{1}{1 + e^{-0.4}} \approx 0.5987$$

### B. Cálculo en la Capa de Salida

**Para la neurona de salida $Y$:**
1. Suma neta ($Z_y$):
   $$Z_y = (A_{h1} \cdot w_{h1}) + (A_{h2} \cdot w_{h2}) + b_y$$
   $$Z_y = (0.6457 \cdot 0.7) + (0.5987 \cdot -0.4) + 0.3$$
   $$Z_y = 0.4520 - 0.2395 + 0.3 = 0.5125$$
2. Activación Final ($A_y$):
   $$A_y = \sigma(0.5125) = \frac{1}{1 + e^{-0.5125}} \approx 0.6254$$

*El valor predicho por la red es **0.6254**, mientras que el valor esperado real es **1**.*

---

## 3. Paso 2: Propagación hacia Atrás (Backpropagation)

Calculamos el error y lo propagamos hacia atrás calculando los gradientes locales ($\delta$).

### A. Gradiente en la Capa de Salida ($\delta_y$)
Multiplicamos el error por la derivada de la función sigmoide en la salida:
$$\delta_y = (T - A_y) \cdot A_y(1 - A_y)$$
$$\delta_y = (1 - 0.6254) \cdot 0.6254 \cdot (1 - 0.6254)$$
$$\delta_y = 0.3746 \cdot 0.6254 \cdot 0.3746 \approx 0.0877$$

### B. Gradientes en la Capa Oculta ($\delta_{h}$)
El gradiente de cada neurona oculta depende del gradiente de la salida filtrado por su respectivo peso de conexión y la derivada de su propia activación.

**Para $H_1$:**
$$\delta_{h1} = (\delta_y \cdot w_{h1}) \cdot A_{h1}(1 - A_{h1})$$
$$\delta_{h1} = (0.0877 \cdot 0.7) \cdot 0.6457 \cdot (1 - 0.6457)$$
$$\delta_{h1} = 0.0614 \cdot 0.6457 \cdot 0.3543 \approx 0.0140$$

**Para $H_2$:**
$$\delta_{h2} = (\delta_y \cdot w_{h2}) \cdot A_{h2}(1 - A_{h2})$$
$$\delta_{h2} = (0.0877 \cdot -0.4) \cdot 0.5987 \cdot (1 - 0.5987)$$
$$\delta_{h2} = -0.0351 \cdot 0.5987 \cdot 0.4013 \approx -0.0084$$

---

## 4. Paso 3: Actualización de Pesos y Biases

Aplicamos la regla del gradiente descendiente con una tasa de aprendizaje $\alpha = 0.25$:
$$\text{Peso}_{\text{nuevo}} = \text{Peso}_{\text{anterior}} + (\alpha \cdot \delta \cdot \text{Entrada})$$

### A. Ajuste de Pesos de Capa Oculta a Salida
* $w_{h1} = 0.7 + (0.25 \cdot 0.0877 \cdot 0.6457) = 0.7 + 0.0142 = \mathbf{0.7142}$
* $w_{h2} = -0.4 + (0.25 \cdot 0.0877 \cdot 0.5987) = -0.4 + 0.0131 = \mathbf{-0.3869}$
* $b_y = 0.3 + (0.25 \cdot 0.0877 \cdot 1) = 0.3 + 0.0219 = \mathbf{0.3219}$

### B. Ajuste de Pesos de Entrada a Capa Oculta
Recordar que para este ejemplo $X_1 = 1$ y $X_2 = 0$.

**Hacia Neurona $H_1$:**
* $w_{11} = 0.5 + (0.25 \cdot 0.0140 \cdot 1) = \mathbf{0.5035}$
* $w_{21} = 0.3 + (0.25 \cdot 0.0140 \cdot 0) = \mathbf{0.3000}$
* $b_{h1} = 0.1 + (0.25 \cdot 0.0140 \cdot 1) = \mathbf{0.1035}$

**Hacia Neurona $H_2$:**
* $w_{12} = 0.2 + (0.25 \cdot -0.0084 \cdot 1) = \mathbf{0.1979}$
* $w_{22} = 0.4 + (0.25 \cdot -0.0084 \cdot 0) = \mathbf{0.4000}$
* $b_{h2} = 0.2 + (0.25 \cdot -0.0084 \cdot 1) = \mathbf{0.1979}$

Tras esta primera iteración, los pesos se han corregido en la dirección correcta para reducir la función de pérdida.

---

## 5. Código de Verificación en Python (OpenCode)

Puedes ejecutar el siguiente script directamente en tu entorno para automatizar y verificar el entrenamiento completo sobre toda la tabla de verdad de la compuerta XOR:

```python
import numpy as np

# 1. Funciones Base
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Datos del problema XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 3. Inicialización de Hiperparámetros
alpha = 0.25
epochs = 20000  # Número alto de iteraciones para lograr la convergencia

# Fijar pesos fijos para validación reproducible
np.random.seed(42)
wh = np.random.uniform(size=(2, 2))
bh = np.random.uniform(size=(1, 2))
wout = np.random.uniform(size=(2, 1))
bout = np.random.uniform(size=(1, 1))

print("--- ENTRENANDO LA RED MULTICAPA ---")
for epoch in range(epochs):
    # Forward Propagation (Lote completo)
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activations = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_activations, wout) + bout
    output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    error_hidden_layer = d_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activations)
    
    # Actualización de pesos y sesgos
    wout += hidden_layer_activations.T.dot(d_output) * alpha
    bout += np.sum(d_output, axis=0, keepdims=True) * alpha
    wh += X.T.dot(d_hidden_layer) * alpha
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * alpha

print("\nResultados tras el entrenamiento:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} -> Predicción: {output[i].item():.4f} -> Esperado: {y[i][0]}")

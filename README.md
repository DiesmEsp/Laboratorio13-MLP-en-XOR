# Laboratorio 13: Resolucion de XOR con un MLP

Este repositorio presenta una tarea orientada a implementar un perceptron multicapa capaz de aprender la compuerta logica XOR, uno de los ejemplos clasicos para introducir redes neuronales no lineales.

## Contexto de la tarea

El problema XOR no puede resolverse correctamente con un perceptron simple porque no es linealmente separable. Por ello, la actividad plantea construir una red neuronal con una capa oculta y aplicar el proceso completo de aprendizaje:

* propagacion hacia adelante
* calculo del error
* backpropagation
* actualizacion de pesos y bias

## Objetivo general

Desarrollar una red neuronal `2-2-1` que reciba dos entradas binarias, procese la informacion en una capa oculta de dos neuronas y produzca una salida capaz de aproximar la tabla de verdad XOR.

## Objetivos especificos

* Comprender por que XOR requiere una arquitectura multicapa.
* Aplicar la funcion sigmoide como activacion en la red.
* Desarrollar al menos una iteracion manual de forward propagation y backpropagation.
* Implementar la solucion completa en Python o en Jupyter Notebook.
* Verificar que la red aprenda las salidas esperadas `0, 1, 1, 0`.

## Alcance esperado

La entrega puede orientarse de dos formas:

* como script `.py`, priorizando una solucion tecnica directa y reproducible
* como notebook `.ipynb`, priorizando la explicacion paso a paso del procedimiento

## Resultado esperado

Al finalizar la tarea, se espera contar con una implementacion funcional del MLP, una explicacion clara del proceso de entrenamiento y evidencia de que la red logra modelar correctamente la compuerta XOR.

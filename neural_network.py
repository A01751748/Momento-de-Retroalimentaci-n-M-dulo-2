import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def layer(A_prev, W, b):
    """Calcula Z de una capa (W @ A_prev + b).

    Args:
        A_prev (np.ndarray): activación de la capa anterior, shape (n_prev, m).
        W (np.ndarray): pesos de la capa, shape (n_curr, n_prev).
        b (np.ndarray): bias de la capa, shape (n_curr, 1).

    Returns:
        np.ndarray: Z de la capa, shape (n_curr, m).
    """
    return W @ A_prev + b

def sigmoid(Z):
    """Aplica sigmoid elemento por elemento.

    Args:
        Z (np.ndarray): valores de entrada.

    Returns:
        np.ndarray: valores entre 0 y 1, mismo shape que Z.
    """
    return 1/(1+np.exp(-Z))

def relu(Z):
    """Aplica ReLU elemento por elemento (max(0, Z)).

    Args:
        Z (np.ndarray): valores de entrada.

    Returns:
        np.ndarray: mismo shape que Z, con los negativos en 0.
    """
    return np.maximum(0,Z)

def feed_forward_one(A_prev, W, b, activation):
    """Calcula Z y la activación A de una sola capa.

    Args:
        A_prev (np.ndarray): activación de la capa anterior.
        W (np.ndarray): pesos de la capa.
        b (np.ndarray): bias de la capa.
        activation (str): "relu" o "sigmoid".

    Returns:
        tuple: (Z, A) de esta capa.
    """
    Z = layer(A_prev, W, b)
    if activation == "relu":
        return Z, relu(Z)
    elif activation == "sigmoid":
        return Z, sigmoid(Z)

def inicializar_parametros(architecture):
    """Inicializa pesos (aleatorios) y bias (en cero) para cada capa.

    Args:
        architecture (list[int]): neuronas por capa, incluyendo la entrada.

    Returns:
        tuple: (W, b), listas donde la posición 0 es None (capa de entrada).
    """
    W = []
    b = []
    for i in range(len(architecture)):
        if i==0:
            W.append(None)
            b.append(None)
            continue
        else:
            W.append(np.random.randn(architecture[i], architecture[i-1]))
            b.append(np.zeros((architecture[i],1)))

    return W, b

def feed_forward(X, W, b, activations):
    """Propaga X hacia adelante a través de todas las capas.

    Args:
        X (array-like): entrada, shape (n_features, m).
        W (list): pesos por capa.
        b (list): bias por capa.
        activations (list[str]): activación de cada capa real.

    Returns:
        tuple: (A, A_cache, Z_cache) — salida final y los valores
        intermedios de cada capa, necesarios para el backward pass.
    """
    X = np.array(X)
    A_cache = []
    Z_cache = []
    A = X
    for i in range(len(W)):
        if i == 0:
            continue
        A_cache.append(A)
        Z, A = feed_forward_one(A, W[i], b[i], activations[i-1])
        Z_cache.append(Z)
    return A, A_cache, Z_cache

def binary_cross_entropy(A, Y):
    """Calcula la pérdida promedio de binary cross-entropy.

    Args:
        A (np.ndarray): predicciones de la red, shape (1, m).
        Y (np.ndarray): etiquetas reales, shape (1, m).

    Returns:
        float: pérdida promedio sobre los m ejemplos.
    """
    return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def relu_d(Z):
    """Derivada de ReLU (1 donde Z>0, 0 donde Z<=0).

    Args:
        Z (np.ndarray): valores de entrada (el Z original de la capa).

    Returns:
        np.ndarray: mismo shape que Z.
    """
    return np.where(Z > 0, 1, 0)

def backward_one_layer(m, A, Y, dA, Z, A_prev, W, last_layer):
    """Calcula los gradientes (dW, db, dA_prev) de una sola capa.

    Args:
        m (int): número de ejemplos del batch.
        A (np.ndarray): predicción final de la red (solo si last_layer=True).
        Y (np.ndarray): etiquetas reales (solo si last_layer=True).
        dA (np.ndarray): gradiente que llega de la capa siguiente (solo si last_layer=False).
        Z (np.ndarray): Z de esta capa, calculado en el forward pass.
        A_prev (np.ndarray): entrada de esta capa, del forward pass.
        W (np.ndarray): pesos de esta capa.
        last_layer (bool): True si es la capa de salida.

    Returns:
        tuple: (dW, db, dA_prev) de esta capa.
    """
    if last_layer:
        dZ = (1/m)*(A-Y) # simplificación de sigmoid + cross-entropy en la capa de salida
    else:
        dZ = dA*relu_d(Z) # regla de la cadena: gradiente que llega * derivada de relu
    dW = dZ @ A_prev.T # shape (n_curr, n_prev), igual que W
    db = np.sum(dZ, axis=1, keepdims=True) # suma las contribuciones de los m ejemplos
    dA_prev = W.T @ dZ # gradiente que se propaga hacia la capa anterior

    return dW, db, dA_prev

def backward(A, Y, A_cache, Z_cache, W):
    """Recorre todas las capas de atrás hacia adelante calculando sus gradientes.

    Args:
        A (np.ndarray): predicción final de la red.
        Y (np.ndarray): etiquetas reales.
        A_cache (list): A_prev guardado de cada capa durante el forward.
        Z_cache (list): Z guardado de cada capa durante el forward.
        W (list): pesos de la red.

    Returns:
        tuple: (dW_list, db_list), en el mismo orden que W (sin la capa de entrada).
    """
    dW_list = []
    db_list = []
    m = Y.shape[1]
    for i in range(len(W)-1,0, -1):
        if i == len(W)-1:
            last_layer = True
            dA_prev = None
        else: 
            last_layer = False
        dW, db, dA_prev = backward_one_layer(m, A, Y, dA_prev, Z_cache[i-1], A_cache[i-1], W[i], last_layer)
        dW_list.append(dW)
        db_list.append(db)
    dW_list.reverse()
    db_list.reverse()
    return dW_list, db_list

def actualizar_parametros(W, b, dW_list, db_list, learning_rate):
    """Actualiza W y b restando el gradiente escalado por el learning rate.

    Args:
        W (list): pesos actuales.
        b (list): bias actuales.
        dW_list (list): gradiente de W por capa.
        db_list (list): gradiente de b por capa.
        learning_rate (float): tamaño del paso de actualización.

    Returns:
        tuple: (W, b) actualizados.
    """
    for i in range(len(dW_list)):
        W[i+1] -= learning_rate*dW_list[i]
        b[i+1] -= learning_rate*db_list[i]
    return W, b

def fit(X, Y, epochs, architecture, activation, learning_rate):
    """Entrena la red con gradient descent durante varias épocas.

    Args:
        X (array-like): datos de entrada.
        Y (np.ndarray): etiquetas reales.
        epochs (int): número de iteraciones de entrenamiento.
        architecture (list[int]): neuronas por capa.
        activation (list[str]): activación de cada capa real.
        learning_rate (float): tamaño del paso de gradient descent.

    Returns:
        tuple: (W, b, losses) — parámetros entrenados y el historial de pérdida por época.
    """
    losses = []
    W, b = inicializar_parametros(architecture)
    for i in range(epochs):
        A, A_cache, Z_cache = feed_forward(X, W, b, activation)
        dW_list, db_list = backward(A, Y, A_cache, Z_cache, W)
        W, b = actualizar_parametros(W, b, dW_list, db_list, learning_rate)
        loss = binary_cross_entropy(A, Y)
        losses.append(loss)
        if i%500 == 0:
            print(f"loss: {loss}")

    print(f"Final loss: {loss}")
    return W, b, losses

def metricas_desempeño(Y, A):
    """Calcula e imprime la matriz de confusión y métricas de desempeño.

    Args:
        Y (np.ndarray): etiquetas reales, shape (1, m).
        A (np.ndarray): predicciones de la red (probabilidades), shape (1, m).

    Returns:
        None
    """
    A = np.round(A)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(Y.shape[1]):
        if Y[0][i] == 0 and A[0][i] == 0:
            TN += 1
        elif Y[0][i] == 0 and A[0][i] == 1:
            FP += 1
        elif Y[0][i] == 1 and A[0][i] == 0:
            FN += 1
        elif Y[0][i] == 1 and A[0][i] == 1:
            TP += 1

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F1 = 2*((precision*recall)/(precision+recall))

    matriz_confusion = np.array([
        [TN, FP],
        [FN, TP]
    ])

    print("Matriz de confusión:")
    print(matriz_confusion)

    print(f"\nAccuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1: {F1}")

    return None


if __name__ == "__main__":
    X = [[0, 0, 1, 1],[0, 1, 0, 1]]
    Y = np.array([[0, 1, 1, 0]])
    W, b, losses = fit(X, Y, epochs=3000, architecture=[2, 4, 1], activation=['relu', 'sigmoid'], learning_rate=0.1)
    A, _, _ = feed_forward(X, W, b, activations=['relu', 'sigmoid'])
    print(f"predicciones: {A}")
    metricas_desempeño(Y, A)
    plt.plot(losses)

    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss durante el entrenamiento")

    plt.show()


import numpy as np

class MLP:
    """
    Perceptron multicapa configurable.
    capas_ocultas : lista con el numero de neuronas por capa oculta
                    ej: [128] -> 1 capa oculta de 128 neuronas
                        [128, 64] -> 2 capas ocultas
    optimizador   : 'sgd' | 'momentum' | 'adam'
    """
    def __init__(self, n_entrada, capas_ocultas, n_salida, alpha=0.01, optimizador='sgd'):
        #Determinar el alpha y el optimizador
        self.alpha = alpha
        self.optimizador = optimizador
        #Determinar la capas, entrada : 784 porque cantidad de entrada y salida :10 porque de 0 hasta 9 y cambiamos las capas ocultas
        self.capas = [n_entrada] + capas_ocultas + [n_salida]
        # Inicializacion de Xavier (tecnica para calcular el peso con la sigmoide) para evitar saturacion de la sigmoide
        #Si los pesos son demasiado cerca de 1 o 0 la sigmoide satura imediatamente
        #Xavier calcula una lim+ y lim- para no tener esa saturacion
        self.W = []
        self.b = []
        for i in range(len(self.capas) - 1):
            lim = np.sqrt(6 / (self.capas[i] + self.capas[i+1]))  # Xavier
            self.W.append(np.random.uniform(-lim, lim, (self.capas[i], self.capas[i+1])))
            self.b.append(np.zeros((1, self.capas[i+1])))

        # Estado interno segun el optimizador
        n = len(self.W)
        if optimizador == 'momentum':
            # Velocidad acumulada para cada capa (W y b)
            # beta: cuanto se conserva la velocidad anterior (tipicamente 0.9)
            self.beta   = 0.9
            self.vW = [np.zeros_like(w) for w in self.W]
            self.vb = [np.zeros_like(b) for b in self.b]

        elif optimizador == 'adam':
            # Adam mantiene dos momentos: m (1er orden) y v (2do orden)
            # beta1=0.9, beta2=0.999 son los valores estandar de los autores
            self.beta1  = 0.9
            self.beta2  = 0.999
            self.eps    = 1e-8
            self.t      = 0  # contador de pasos para la correccion de sesgo
            self.mW = [np.zeros_like(w) for w in self.W]
            self.mb = [np.zeros_like(b) for b in self.b]
            self.vW = [np.zeros_like(w) for w in self.W]
            self.vb = [np.zeros_like(b) for b in self.b]

    # ---- Funciones de activacion ----

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Formula de la sigmoid

    def _sigmoid_deriv(self, a):
        return a * (1 - a)  # Formula de la derivada de la sigmoide. Eso es para la backpropagation

    def _softmax(self, z):  # Softmax es la ultima capa y permite de dar porcentage por cada uno de los digits 0 hasta 9
        # Restamos el maximo por estabilidad numerica
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ---- Forward pass ----

    def _forward(self, X):
        activaciones = [X]  # Para stockear cada activacion
        a = X
        for i in range(len(self.W) - 1):  # Para cada capas menos la ultima
            z = a @ self.W[i] + self.b[i]  # Hacemos el producto matricial + el sesgo b
            a = self._sigmoid(z)  # funcion de activacion Sigmoide
            activaciones.append(a)  # Agregamos el a a la activacion
        # Ultima capa: capa de salida con softmax
        z = a @ self.W[-1] + self.b[-1]  # Producto matricial + sesgo, el -1 es para empezar del final
        a = self._softmax(z)  # Permite obtener por cada numero de 0 a 9 la probabilidad que sea ese digit
        activaciones.append(a)
        return activaciones

    # ---- Backpropagation + actualizacion segun optimizador ----
    # Permite cambiar los pesos y hacer que el modelo aprende

    def _backward(self, activaciones, y_onehot):
        m = y_onehot.shape[0]  # Cantidad de raw con el onehotencoder
        deltas = [None] * len(self.W)
        # Error en la capa de salida (softmax + cross-entropy -> delta simple)
        deltas[-1] = activaciones[-1] - y_onehot
        # Propagacion hacia atras por las capas ocultas
        for i in range(len(self.W) - 2, -1, -1):
            deltas[i] = (deltas[i+1] @ self.W[i+1].T) * self._sigmoid_deriv(activaciones[i+1])

        if self.optimizador == 'sgd':
            # SGD: actualizacion directa con el gradiente
            for i in range(len(self.W)):
                grad_W = activaciones[i].T @ deltas[i] / m
                grad_b = deltas[i].mean(axis=0, keepdims=True)
                self.W[i] -= self.alpha * grad_W  # Descenso por gradiente
                self.b[i] -= self.alpha * grad_b

        elif self.optimizador == 'momentum':
            # Momentum: acumula una "velocidad" en la direccion del gradiente
            # Ventaja: acelera el aprendizaje y reduce las oscilaciones
            # v = beta * v_anterior + alpha * gradiente
            # W = W - v
            for i in range(len(self.W)):
                grad_W = activaciones[i].T @ deltas[i] / m
                grad_b = deltas[i].mean(axis=0, keepdims=True)
                self.vW[i] = self.beta * self.vW[i] + self.alpha * grad_W
                self.vb[i] = self.beta * self.vb[i] + self.alpha * grad_b
                self.W[i] -= self.vW[i]
                self.b[i] -= self.vb[i]

        elif self.optimizador == 'adam':
            # Adam: adapta el alpha para cada peso individualmente
            # Combina Momentum (1er momento) + RMSProp (2do momento)
            # Ventaja: converge mas rapido y es mas estable
            self.t += 1
            for i in range(len(self.W)):
                grad_W = activaciones[i].T @ deltas[i] / m
                grad_b = deltas[i].mean(axis=0, keepdims=True)
                # Actualizacion del 1er momento (media del gradiente)
                self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * grad_W
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * grad_b
                # Actualizacion del 2do momento (varianza del gradiente)
                self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * grad_W**2
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * grad_b**2
                # Correccion de sesgo (importantes en los primeros pasos)
                mW_hat = self.mW[i] / (1 - self.beta1**self.t)
                mb_hat = self.mb[i] / (1 - self.beta1**self.t)
                vW_hat = self.vW[i] / (1 - self.beta2**self.t)
                vb_hat = self.vb[i] / (1 - self.beta2**self.t)
                self.W[i] -= self.alpha * mW_hat / (np.sqrt(vW_hat) + self.eps)
                self.b[i] -= self.alpha * mb_hat / (np.sqrt(vb_hat) + self.eps)

    # ---- Entrenamiento ----

    def fit(self, X, y, epochs=50, batch_size=32, X_val=None, y_val=None, verbose=True):
        n_clases = self.capas[-1]
        y_onehot = np.eye(n_clases)[y]  # OneHotEncoder para transformar los valores de 0 a 9
        m = X.shape[0]
        history_train, history_val = [], []

        for epoch in range(1, epochs + 1):
            # Mini-batch shuffle
            idx = np.random.permutation(m)  # Aleatorizamos los digits
            for start in range(0, m, batch_size):  # Creamos los batchs
                batch = idx[start:start + batch_size]
                acts = self._forward(X[batch])
                self._backward(acts, y_onehot[batch])

            acc_train = self.score(X, y)  # Mide la accuracy
            history_train.append(acc_train)  # Guarda la historia del accuracy

            if X_val is not None:
                acc_val = self.score(X_val, y_val)
                history_val.append(acc_val)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                msg = f"  [Epoch {epoch:>3}] train={acc_train*100:.2f}%"
                if X_val is not None:
                    msg += f"  val={acc_val*100:.2f}%"
                print(msg)

        return history_train, history_val

    # ---- Prediccion ----

    def predict(self, X):
        activaciones = self._forward(X)
        return np.argmax(activaciones[-1], axis=1)  # Clase con el porcentaje mas alto

    def score(self, X, y):
        return np.mean(self.predict(X) == y)  # Proporcion de buena respuesta

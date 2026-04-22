import numpy as np

class MLPv2:
    """
    Perceptron multicapa mejorado para alcanzar 98% de exactitud.
    Mejoras respecto a MLPv1:
      - Soporte para activacion ReLU (mejor flujo de gradiente que sigmoid)
      - Decaimiento del learning rate (lr_decay)
      - Optimizadores: sgd, momentum, adam
    """
    def __init__(self, n_entrada, capas_ocultas, n_salida,
                 alpha=0.001, optimizador='adam', activacion='relu', lr_decay=0.0):
        self.alpha       = alpha
        self.alpha_ini   = alpha
        self.optimizador = optimizador
        self.activacion  = activacion
        self.lr_decay    = lr_decay  # reduce alpha cada epoch: alpha = alpha_ini / (1 + decay * epoch)
        self.capas = [n_entrada] + capas_ocultas + [n_salida]

        # Inicializacion de pesos
        self.W, self.b = [], []
        for i in range(len(self.capas) - 1):
            if activacion == 'relu':
                # He initialization: optima para ReLU. ELigimos un numero aleatorio entre O y el sqrt(2/N) con N la cantidad de input en el nodo
                std = np.sqrt(2.0 / self.capas[i])
                self.W.append(np.random.randn(self.capas[i], self.capas[i+1]) * std)
            else:
                # Xavier: optima para sigmoid. Eligimos un numero aleatorio entre 0 y sqrt(6/(n_in +n_out))
                lim = np.sqrt(6 / (self.capas[i] + self.capas[i+1]))
                self.W.append(np.random.uniform(-lim, lim, (self.capas[i], self.capas[i+1])))
            self.b.append(np.zeros((1, self.capas[i+1])))

        # Variables internas del optimizador
        n = len(self.W)
        if optimizador == 'momentum': #Momentum es un optimizador en cual agregamos un termino (momentum factor between 0 y 1) que permite hacer un smooth en la optimizacion en fonction de paso de antes
            self.beta = 0.9 #Creamos el componente momentum
            self.vW = [np.zeros_like(w) for w in self.W] #Creamos la matriz de peso
            self.vb = [np.zeros_like(b) for b in self.b] #creamos la matriz de biais 
        elif optimizador == 'adam':
            #beta1 controla el promedio de gradientes (90% direccion pasada+10% nueva),
            #beta2 controla la varianza de gradientes, cerca de 1 mucha memoria de los gradientes pasados 
            #eps permite no tener divison por O 
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8 
            self.t = 0 #cantidad de iteracion
            self.mW = [np.zeros_like(w) for w in self.W] #Creamos la matriz de pesos
            self.mb = [np.zeros_like(b) for b in self.b] #Creamos la matriz de biais
            self.vW = [np.zeros_like(w) for w in self.W] #Creamos la matriz de viarianza de peso
            self.vb = [np.zeros_like(b) for b in self.b] #Creamos la matriz de varianza de biais

    # ---- Funciones de activacion ----

    def _relu(self, z): #Funcion de activacion ReLu
        # ReLU: max(0, z) - no satura para valores grandes, mejor gradiente
        return np.maximum(0, z)

    def _relu_deriv(self, z): #Derivada para relu
        return (z > 0).astype(float)

    def _sigmoid(self, z): #Funcion de activacion sigmoide
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, a): #Derivada para calcular el gradiente
        return a * (1 - a)

    def _softmax(self, z): #Ultima capa que permite determinar el porcentaje por cada uno de los numeros
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _activar(self, z): #eligir la funcion de activacion en el nodo
        return self._relu(z) if self.activacion == 'relu' else self._sigmoid(z)

    def _activar_deriv(self, z_o_a):
        return self._relu_deriv(z_o_a) if self.activacion == 'relu' else self._sigmoid_deriv(z_o_a)

    # ---- Forward ----

    def _forward(self, X):
        # Guardamos tanto z (pre-activacion) como a (post-activacion) para ReLU
        self._zs = []
        activaciones = [X] # Para stockear cada activacion
        a = X
        for i in range(len(self.W) - 1):  # Para cada capas menos la ultima
            z = a @ self.W[i] + self.b[i] # Hacemos el producto matricial + el sesgo b
            self._zs.append(z) #Valor pre activacion para la ReLu
            a = self._activar(z) #Puede ser Sigmoid o Relu
            activaciones.append(a)
        z = a @ self.W[-1] + self.b[-1] # Producto matricial + sesgo, el -1 es para empezar del final
        self._zs.append(z) 
        activaciones.append(self._softmax(z))# Permite obtener por cada numero de 0 a 9 la probabilidad que sea ese digit
        return activaciones

    # ---- Backward ----

    def _backward(self, activaciones, y_onehot):
        m = y_onehot.shape[0] # Cantidad de raw con el onehotencoder
        deltas = [None] * len(self.W)
        # Error en la capa de salida (softmax + cross-entropy -> delta simple)
        deltas[-1] = activaciones[-1] - y_onehot
        # Propagacion hacia atras por las capas ocultas
        for i in range(len(self.W) - 2, -1, -1):
            # Para ReLU usamos z pre-activacion, para sigmoid usamos a post-activacion
            if self.activacion == 'relu':
                deriv = self._activar_deriv(self._zs[i])
            else:
                deriv = self._activar_deriv(activaciones[i+1])
            deltas[i] = (deltas[i+1] @ self.W[i+1].T) * deriv
        self._actualizar_pesos(activaciones, deltas, m)

    def _actualizar_pesos(self, activaciones, deltas, m):
        # Tasa de aprendizaje local para evitar acceder repetidamente al atributo
        alpha = self.alpha
        # Recorrer cada capa de la red (una matriz de pesos W[i] y un bias b[i] por capa)
        for i in range(len(self.W)):
            # Gradiente de la pérdida respecto a W[i]:
            # activaciones[i].T tiene forma (n_entradas, m), deltas[i] tiene forma (m, n_salidas)
            # El producto matricial da (n_entradas, n_salidas), normalizado por m para obtener el promedio del batch
            gW = activaciones[i].T @ deltas[i] / m
            # Gradiente respecto al bias: promedio de los deltas sobre los ejemplos del batch
            gb = deltas[i].mean(axis=0, keepdims=True)

            if self.optimizador == 'sgd':
                # SGD estándar: descenso de gradiente puro, sin acumulación de historia
                self.W[i] -= alpha * gW
                self.b[i] -= alpha * gb

            elif self.optimizador == 'momentum':
                # Momentum: acumula una velocidad exponencialmente ponderada del gradiente
                # v_t = beta * v_{t-1} + alpha * g_t  (beta controla el «olvido» de la historia)
                self.vW[i] = self.beta * self.vW[i] + alpha * gW
                self.vb[i] = self.beta * self.vb[i] + alpha * gb
                # El peso se actualiza con la velocidad acumulada, no directamente con el gradiente
                self.W[i] -= self.vW[i]
                self.b[i] -= self.vb[i]

            elif self.optimizador == 'adam':
                # Incrementar el contador de pasos (necesario para la corrección de sesgo)
                self.t += 1

                # Primer momento (media móvil exponencial del gradiente) — «dirección»
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * gW
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb

                # Segundo momento (media móvil exponencial del gradiente al cuadrado) — «magnitud»
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t²
                self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * gW**2
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * gb**2

                # Corrección de sesgo: los momentos se inicializan a 0, por lo que están sesgados
                # hacia 0 al inicio del entrenamiento; dividir por (1 - beta^t) corrige esto
                mW_h = self.mW[i] / (1 - self.beta1**self.t)
                mb_h = self.mb[i] / (1 - self.beta1**self.t)
                vW_h = self.vW[i] / (1 - self.beta2**self.t)
                vb_h = self.vb[i] / (1 - self.beta2**self.t)

                # Actualización: el paso se normaliza por la raíz del segundo momento corregido
                # (+eps evita la división por cero cuando el gradiente es muy pequeño)
                self.W[i] -= alpha * mW_h / (np.sqrt(vW_h) + self.eps)
                self.b[i] -= alpha * mb_h / (np.sqrt(vb_h) + self.eps)

    # ---- Entrenamiento ----

    def fit(self, X, y, epochs=100, batch_size=64, X_val=None, y_val=None, verbose=True,
            paciencia=None, min_delta=0.001):
        """
        paciencia  : numero de epochs consecutivas sin mejorar min_delta antes de parar.
                     None desactiva el early stopping.
        min_delta  : mejora minima (en fraccion, ej. 0.001 = 0.1%) para contar como progreso.
        """
        n_clases  = self.capas[-1]
        y_onehot  = np.eye(n_clases)[y]
        m = X.shape[0]
        history_train, history_val = [], []

        # --- Estado del early stopping ---
        mejor_acc_val   = -np.inf   # mejor exactitud de validacion vista hasta ahora
        epochs_sin_mejora = 0       # contador de epochs consecutivas sin superar el umbral

        for epoch in range(1, epochs + 1):

            # --- Decaimiento de la tasa de aprendizaje ---
            # Formula: alpha = alpha_ini / (1 + lr_decay * epoch)
            if self.lr_decay > 0:
                self.alpha = self.alpha_ini / (1 + self.lr_decay * epoch)

            # --- Mini-batch estocastico ---
            idx = np.random.permutation(m)
            for start in range(0, m, batch_size):
                batch = idx[start:start + batch_size]
                acts  = self._forward(X[batch])
                self._backward(acts, y_onehot[batch])

            # --- Evaluacion tras cada epoch ---
            acc_train = self.score(X, y)
            history_train.append(acc_train)

            if X_val is not None:
                acc_val = self.score(X_val, y_val)
                history_val.append(acc_val)

            # --- Visualizacion periodica ---
            if verbose and (epoch % 10 == 0 or epoch == 1):
                msg = f"  [Epoch {epoch:>3}] train={acc_train*100:.2f}%"
                if X_val is not None:
                    msg += f"  val={acc_val*100:.2f}%"
                if self.lr_decay > 0:
                    msg += f"  lr={self.alpha:.5f}"
                print(msg)

            # --- Early stopping: solo activo si hay datos de validacion y paciencia definida ---
            if paciencia is not None and X_val is not None:
                if acc_val >= mejor_acc_val + min_delta:
                    # Mejora suficiente: reiniciamos el contador
                    mejor_acc_val     = acc_val
                    epochs_sin_mejora = 0
                else:
                    epochs_sin_mejora += 1
                    if epochs_sin_mejora >= paciencia:
                        if verbose:
                            print(f"  [Early stopping] epoch {epoch} — sin mejora de {min_delta*100:.1f}% "
                                  f"durante {paciencia} epochs (mejor val: {mejor_acc_val*100:.2f}%)")
                        break

        return history_train, history_val

    def predict(self, X):
        return np.argmax(self._forward(X)[-1], axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    # ---- Persistencia del modelo ----

    def guardar(self, ruta):
        """Guarda pesos, sesgos, hiperparametros y estado del optimizador en un .npz"""
        datos = {
            'capas':      np.array(self.capas),
            'alpha_ini':  np.array(self.alpha_ini),
            'lr_decay':   np.array(self.lr_decay),
            'activacion': np.array(self.activacion),
            'optimizador':np.array(self.optimizador),
            't':          np.array(getattr(self, 't', 0)),
        }
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            datos[f'W_{i}'] = w
            datos[f'b_{i}'] = b
        if self.optimizador == 'adam':
            for i in range(len(self.W)):
                datos[f'mW_{i}'] = self.mW[i]
                datos[f'mb_{i}'] = self.mb[i]
                datos[f'vW_{i}'] = self.vW[i]
                datos[f'vb_{i}'] = self.vb[i]
        elif self.optimizador == 'momentum':
            for i in range(len(self.W)):
                datos[f'vW_{i}'] = self.vW[i]
                datos[f'vb_{i}'] = self.vb[i]
        np.savez(ruta, **datos)

    @classmethod
    def cargar(cls, ruta):
        """Reconstruye un modelo completo desde un archivo .npz guardado con guardar()"""
        d = np.load(ruta, allow_pickle=True)
        capas      = d['capas'].tolist()
        optimizador = str(d['optimizador'])
        activacion  = str(d['activacion'])
        n_capas    = len(capas) - 1

        modelo = cls(
            n_entrada=capas[0], capas_ocultas=capas[1:-1], n_salida=capas[-1],
            alpha=float(d['alpha_ini']), optimizador=optimizador,
            activacion=activacion, lr_decay=float(d['lr_decay']),
        )
        modelo.W = [d[f'W_{i}'] for i in range(n_capas)]
        modelo.b = [d[f'b_{i}'] for i in range(n_capas)]
        if optimizador == 'adam':
            modelo.t  = int(d['t'])
            modelo.mW = [d[f'mW_{i}'] for i in range(n_capas)]
            modelo.mb = [d[f'mb_{i}'] for i in range(n_capas)]
            modelo.vW = [d[f'vW_{i}'] for i in range(n_capas)]
            modelo.vb = [d[f'vb_{i}'] for i in range(n_capas)]
        elif optimizador == 'momentum':
            modelo.vW = [d[f'vW_{i}'] for i in range(n_capas)]
            modelo.vb = [d[f'vb_{i}'] for i in range(n_capas)]
        return modelo

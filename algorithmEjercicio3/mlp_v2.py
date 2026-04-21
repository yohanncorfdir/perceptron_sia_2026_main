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
                # He initialization: optima para ReLU
                std = np.sqrt(2.0 / self.capas[i])
                self.W.append(np.random.randn(self.capas[i], self.capas[i+1]) * std)
            else:
                # Xavier: optima para sigmoid
                lim = np.sqrt(6 / (self.capas[i] + self.capas[i+1]))
                self.W.append(np.random.uniform(-lim, lim, (self.capas[i], self.capas[i+1])))
            self.b.append(np.zeros((1, self.capas[i+1])))

        # Variables internas del optimizador
        n = len(self.W)
        if optimizador == 'momentum':
            self.beta = 0.9
            self.vW = [np.zeros_like(w) for w in self.W]
            self.vb = [np.zeros_like(b) for b in self.b]
        elif optimizador == 'adam':
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
            self.t = 0
            self.mW = [np.zeros_like(w) for w in self.W]
            self.mb = [np.zeros_like(b) for b in self.b]
            self.vW = [np.zeros_like(w) for w in self.W]
            self.vb = [np.zeros_like(b) for b in self.b]

    # ---- Funciones de activacion ----

    def _relu(self, z):
        # ReLU: max(0, z) - no satura para valores grandes, mejor gradiente
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def _softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _activar(self, z):
        return self._relu(z) if self.activacion == 'relu' else self._sigmoid(z)

    def _activar_deriv(self, z_o_a):
        return self._relu_deriv(z_o_a) if self.activacion == 'relu' else self._sigmoid_deriv(z_o_a)

    # ---- Forward ----

    def _forward(self, X):
        # Guardamos tanto z (pre-activacion) como a (post-activacion) para ReLU
        self._zs = []
        activaciones = [X]
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            self._zs.append(z)
            a = self._activar(z)
            activaciones.append(a)
        z = a @ self.W[-1] + self.b[-1]
        self._zs.append(z)
        activaciones.append(self._softmax(z))
        return activaciones

    # ---- Backward ----

    def _backward(self, activaciones, y_onehot):
        m = y_onehot.shape[0]
        deltas = [None] * len(self.W)
        deltas[-1] = activaciones[-1] - y_onehot
        for i in range(len(self.W) - 2, -1, -1):
            # Para ReLU usamos z pre-activacion, para sigmoid usamos a post-activacion
            if self.activacion == 'relu':
                deriv = self._activar_deriv(self._zs[i])
            else:
                deriv = self._activar_deriv(activaciones[i+1])
            deltas[i] = (deltas[i+1] @ self.W[i+1].T) * deriv
        self._actualizar_pesos(activaciones, deltas, m)

    def _actualizar_pesos(self, activaciones, deltas, m):
        alpha = self.alpha
        for i in range(len(self.W)):
            gW = activaciones[i].T @ deltas[i] / m
            gb = deltas[i].mean(axis=0, keepdims=True)

            if self.optimizador == 'sgd':
                self.W[i] -= alpha * gW
                self.b[i] -= alpha * gb

            elif self.optimizador == 'momentum':
                self.vW[i] = self.beta * self.vW[i] + alpha * gW
                self.vb[i] = self.beta * self.vb[i] + alpha * gb
                self.W[i] -= self.vW[i]
                self.b[i] -= self.vb[i]

            elif self.optimizador == 'adam':
                self.t += 1
                self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * gW
                self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb
                self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * gW**2
                self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * gb**2
                mW_h = self.mW[i] / (1 - self.beta1**self.t)
                mb_h = self.mb[i] / (1 - self.beta1**self.t)
                vW_h = self.vW[i] / (1 - self.beta2**self.t)
                vb_h = self.vb[i] / (1 - self.beta2**self.t)
                self.W[i] -= alpha * mW_h / (np.sqrt(vW_h) + self.eps)
                self.b[i] -= alpha * mb_h / (np.sqrt(vb_h) + self.eps)

    # ---- Entrenamiento ----

    def fit(self, X, y, epochs=100, batch_size=64, X_val=None, y_val=None, verbose=True):
        n_clases  = self.capas[-1]
        y_onehot  = np.eye(n_clases)[y]
        m         = X.shape[0]
        history_train, history_val = [], []

        for epoch in range(1, epochs + 1):
            # Decaimiento del learning rate
            if self.lr_decay > 0:
                self.alpha = self.alpha_ini / (1 + self.lr_decay * epoch)

            idx = np.random.permutation(m)
            for start in range(0, m, batch_size):
                batch = idx[start:start + batch_size]
                acts  = self._forward(X[batch])
                self._backward(acts, y_onehot[batch])

            acc_train = self.score(X, y)
            history_train.append(acc_train)
            if X_val is not None:
                acc_val = self.score(X_val, y_val)
                history_val.append(acc_val)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                msg = f"  [Epoch {epoch:>3}] train={acc_train*100:.2f}%"
                if X_val is not None:
                    msg += f"  val={acc_val*100:.2f}%"
                if self.lr_decay > 0:
                    msg += f"  lr={self.alpha:.5f}"
                print(msg)

        return history_train, history_val

    def predict(self, X):
        return np.argmax(self._forward(X)[-1], axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

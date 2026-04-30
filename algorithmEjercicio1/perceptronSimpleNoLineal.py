import numpy as np

class PerceptronNoLineal:
	def __init__(self, N, alpha=0.1):
		# Inicializamos los pesos aleatoriamente y guardamos la tasa de aprendizaje
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def sigmoid(self, x):
		# Funcion de activacion no lineal: retorna un valor continuo entre 0 y 1
		return 1 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		# Derivada de la sigmoide, necesaria para el gradiente descendente
		return x * (1 - x)

	def fit(self, X, y, epochs=20, X_val=None, y_val=None):
		# Agregamos el sesgo (bias) como ultima columna de X
		X = np.c_[X, np.ones((X.shape[0]))]
		if X_val is not None:
			X_val_b = np.c_[X_val, np.ones((X_val.shape[0]))]
		history_train = []
		history_val = []
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				output = self.sigmoid(np.dot(x, self.W))
				error = output - target
				# Actualizamos los pesos usando el gradiente de la funcion de perdida
				self.W += -self.alpha * error * self.sigmoid_deriv(output) * x
			# Exactitud sobre el conjunto de entrenamiento
			preds_train = (self.sigmoid(X.dot(self.W)) >= 0.5).astype(int)
			history_train.append(np.mean(preds_train == y))
			# Exactitud sobre el conjunto de validacion (si se proporciona)
			if X_val is not None:
				preds_val = (self.sigmoid(X_val_b.dot(self.W)) >= 0.5).astype(int)
				history_val.append(np.mean(preds_val == y_val))
		return history_train, history_val

	def predict_proba(self, X, addBias=True):
		# Retorna la probabilidad cruda de la sigmoide (sin umbral)
		X = np.atleast_2d(X)
		if addBias:
			X = np.c_[X, np.ones((X.shape[0]))]
		return self.sigmoid(X.dot(self.W))

	def predict(self, X, threshold=0.4, addBias=True):
		# Aplicamos el umbral configurable para obtener una prediccion binaria
		return (self.predict_proba(X, addBias=addBias) >= threshold).astype(int)

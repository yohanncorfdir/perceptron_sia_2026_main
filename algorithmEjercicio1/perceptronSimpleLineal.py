import numpy as np

class PerceptronLineal:
	def __init__(self, N, alpha=0.1):
		# Inicializamos los pesos aleatoriamente y guardamos la tasa de aprendizaje
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		# Funcion de activacion lineal: retorna 0 o 1
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs=20, X_val=None, y_val=None):
		# Agregamos el sesgo (bias) como ultima columna de X
		X = np.c_[X, np.ones((X.shape[0]))]
		if X_val is not None:
			X_val_b = np.c_[X_val, np.ones((X_val.shape[0]))]
		history_train = []
		history_val = []
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				p = self.step(np.dot(x, self.W))
				# Solo actualizamos los pesos si la prediccion es incorrecta
				if p != target:
					error = p - target
					self.W += -self.alpha * error * x
			# Exactitud sobre el conjunto de entrenamiento
			preds_train = np.array([self.step(np.dot(x, self.W)) for x in X])
			history_train.append(np.mean(preds_train == y))
			# Exactitud sobre el conjunto de validacion (si se proporciona)
			if X_val is not None:
				preds_val = np.array([self.step(np.dot(x, self.W)) for x in X_val_b])
				history_val.append(np.mean(preds_val == y_val))
		return history_train, history_val

	def predict(self, X, addBias=True):
		X = np.atleast_2d(X)
		if addBias:
			X = np.c_[X, np.ones((X.shape[0]))]
		return self.step(np.dot(X, self.W))

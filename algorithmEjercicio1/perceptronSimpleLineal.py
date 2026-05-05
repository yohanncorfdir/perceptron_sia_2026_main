import numpy as np

class PerceptronLineal:
	def __init__(self, N, alpha=0.1):
		# Inicializamos los pesos aleatoriamente y guardamos la tasa de aprendizaje
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def activation(self, x):
		# Funcion de activacion lineal: f(z) = z (identidad)
		return x

	def fit(self, X, y, epochs=20, X_val=None, y_val=None):
		# Agregamos el sesgo (bias) como ultima columna de X
		X = np.c_[X, np.ones((X.shape[0]))]
		if X_val is not None:
			X_val_b = np.c_[X_val, np.ones((X_val.shape[0]))]
		
		history_train = []
		history_val = []
		
		for epoch in np.arange(0, epochs):
			# Barajamos los datos en cada epoch para un entrenamiento mas estable (SGD)
			indices = np.arange(len(X))
			np.random.shuffle(indices)
			
			for i in indices:
				xi = X[i]
				target = y[i]
				# Forward pass
				p = self.activation(np.dot(xi, self.W))
				# Update rule for MSE: W = W - alpha * (p - target) * xi
				error = p - target
				self.W += -self.alpha * error * xi
			
			# Calculo de MSE sobre el conjunto de entrenamiento
			preds_train = self.activation(np.dot(X, self.W))
			mse_train = np.mean((preds_train - y)**2)
			history_train.append(mse_train)
			
			# Calculo de MSE sobre el conjunto de validacion (si se proporciona)
			if X_val is not None:
				preds_val = self.activation(np.dot(X_val_b, self.W))
				mse_val = np.mean((preds_val - y_val)**2)
				history_val.append(mse_val)
				
		return history_train, history_val

	def predict_proba(self, X, addBias=True):
		# Retorna la salida lineal continua
		X = np.atleast_2d(X)
		if addBias:
			X = np.c_[X, np.ones((X.shape[0]))]
		return self.activation(np.dot(X, self.W))

	def predict(self, X, threshold=0.5, addBias=True):
		# Aplicamos un umbral para obtener una prediccion binaria
		return (self.predict_proba(X, addBias=addBias) >= threshold).astype(int)

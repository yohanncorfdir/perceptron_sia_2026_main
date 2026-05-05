import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from perceptronSimpleLineal import PerceptronLineal
from perceptronSimpleNoLineal import PerceptronNoLineal

# --- Carga y normalizacion de TODAS las muestras ---
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/fraud_dataset.csv'))

# Usamos big_model_fraud_probability como target para Knowledge Distillation
# El objetivo es que TinyModel aprenda a replicar la salida de BigModel
X = df.drop(columns=['flagged_fraud','big_model_fraud_probability']).values.astype(float)
y = df['big_model_fraud_probability'].values.astype(float)
y_binary = df['flagged_fraud'].values.astype(int)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

EPOCHS = 20
print(f"Conjunto de datos: {len(X)} muestras, {X.shape[1]} variables de entrada")
print(f"Target: Knowledge Distillation (replicar BigModel)")
print(f"Media de probabilidad de BigModel: {y.mean():.4f}")


# Estudio del potencial de aprendizaje (todas las muestras)
pl  = PerceptronLineal(X.shape[1], alpha=0.1)
pnl = PerceptronNoLineal(X.shape[1], alpha=0.1)

hist_lin,  _ = pl.fit(X, y, epochs=EPOCHS)
hist_nolin, _ = pnl.fit(X, y, epochs=EPOCHS)

print(f"\n{'Epoch':<8} {'Lineal (MSE)':<22} {'No Lineal (MSE)':<22}")


for i, (mse_lin, mse_nolin) in enumerate(zip(hist_lin, hist_nolin), 1):
    print(f"{i:<8} {mse_lin:<22.6f} {mse_nolin:<22.6f}")

print(f"\nMSE Final Lineal:    {hist_lin[-1]:.6f}")
print(f"MSE Final No Lineal: {hist_nolin[-1]:.6f}")

# --- Analisis: underfitting y saturacion ---

# Underfitting: MSE alto incluso sobre los datos de entrenamiento
umbral_underfitting_mse = 0.05
for nombre, hist in [("Lineal", hist_lin), ("No Lineal", hist_nolin)]:
    if min(hist) > umbral_underfitting_mse:
        print(f"{nombre}: UNDERFITTING detectado (MSE minimo = {min(hist):.6f})")
    else:
        print(f"{nombre}: Sin underfitting (MSE minimo = {min(hist):.6f})")

# Saturacion: el MSE deja de mejorar
for nombre, hist in [("Lineal", hist_lin), ("No Lineal", hist_nolin)]:
    mejora = hist[0] - min(hist)
    ultimas = hist[-5:]
    rango_final = max(ultimas) - min(ultimas)
    if rango_final < 0.0001:
        print(f"{nombre}: SATURACION detectada (variacion en ultimas 5 epochs = {rango_final:.6f})")
    else:
        print(f"{nombre}: Sin saturacion (progresion MSE {hist[0]:.6f} -> {hist[-1]:.6f})")

# Seleccion del modelo para estudio de generalizacion
mejor = "No Lineal" if hist_nolin[-1] <= hist_lin[-1] else "Lineal"
print(f"Modelo seleccionado para generalizacion: {mejor}")


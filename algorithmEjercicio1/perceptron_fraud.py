import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from perceptronSimpleLineal import PerceptronLineal
from perceptronSimpleNoLineal import PerceptronNoLineal

# --- Carga y normalizacion de TODAS las muestras ---
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/fraud_dataset.csv'))

#Sacamos flagged_fraud and big_model_fraud_probability, el primero porque es el feature que queremos predecir y la segunda porque da informacion sobre la feature que queremos precedir con bigmodel
X = df.drop(columns=['flagged_fraud','big_model_fraud_probability']).values.astype(float)
y = df['flagged_fraud'].values.astype(int)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

EPOCHS = 20
print(f"Conjunto de datos: {len(X)} muestras, {X.shape[1]} variables de entrada")
print(f"Fraudes: {y.sum()} ({y.mean()*100:.1f}%) | No fraudes: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")


# Estudio del potencial de aprendizaje (todas las muestras)
pl  = PerceptronLineal(X.shape[1], alpha=0.1)
pnl = PerceptronNoLineal(X.shape[1], alpha=0.1)

hist_lin,  _ = pl.fit(X, y, epochs=EPOCHS)
hist_nolin, _ = pnl.fit(X, y, epochs=EPOCHS)

print(f"\n{'Epoch':<8} {'Lineal (acc %)':<22} {'No Lineal (acc %)':<22}")


for i, (a_lin, a_nolin) in enumerate(zip(hist_lin, hist_nolin), 1):
    print(f"{i:<8} {a_lin*100:<22.2f} {a_nolin*100:<22.2f}")

print(f"\nExactitud Lineal:    {hist_lin[-1]*100:.2f}%")
print(f"Exactitud No Lineal: {hist_nolin[-1]*100:.2f}%")

# --- Analisis: underfitting y saturacion ---

# Underfitting: exactitud baja incluso sobre los datos de entrenamiento
umbral_underfitting = 0.80
for nombre, hist in [("Lineal", hist_lin), ("No Lineal", hist_nolin)]:
    if max(hist) < umbral_underfitting:
        print(f"{nombre}: UNDERFITTING detectado (exactitud maxima = {max(hist)*100:.2f}%)")
    else:
        print(f"{nombre}: Sin underfitting (exactitud maxima = {max(hist)*100:.2f}%)")

# Saturacion: la exactitud deja de mejorar
for nombre, hist in [("Lineal", hist_lin), ("No Lineal", hist_nolin)]:
    mejora = max(hist) - hist[0]
    ultimas = hist[-5:]
    rango_final = max(ultimas) - min(ultimas)
    if rango_final < 0.005:
        print(f"{nombre}: SATURACION detectada (variacion en ultimas 5 epochs = {rango_final*100:.3f}%)")
    else:
        print(f"{nombre}: Sin saturacion (progresion {hist[0]*100:.2f}% -> {hist[-1]*100:.2f}%)")

# Seleccion del modelo para estudio de generalizacion
mejor = "No Lineal" if hist_nolin[-1] >= hist_lin[-1] else "Lineal"
print(f"Modelo seleccionado para generalizacion: {mejor}")


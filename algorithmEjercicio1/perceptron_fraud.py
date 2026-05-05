import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from perceptronSimpleLineal import PerceptronLineal
from perceptronSimpleNoLineal import PerceptronNoLineal
import plots

# --- Carga y normalizacion de TODAS las muestras ---
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/fraud_dataset.csv'))

# Usamos big_model_fraud_probability como target para Knowledge Distillation
# El objetivo es que TinyModel aprenda a replicar la salida de BigModel
X = df.drop(columns=['flagged_fraud','big_model_fraud_probability']).values.astype(float)
y = df['big_model_fraud_probability'].values.astype(float)
y_binary = df['flagged_fraud'].values.astype(int)

# Preprocesamiento: Min-Max Scaling
# Normalizamos los datos de entrada al rango [0, 1] para mejorar la estabilidad
# y convergencia de los gradientes en el perceptrón.
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min + 1e-8)

EPOCHS = 30
print(f"Conjunto de datos: {len(X)} muestras, {X.shape[1]} variables de entrada")
print(f"Target: Knowledge Distillation (replicar BigModel)")
print(f"Media de probabilidad de BigModel: {y.mean():.4f}")

# --- Evaluación de la calidad del BigModel ---
# Calculamos R2 y AUC para ver qué tan bien BigModel representa el fraude real
r2_big = plots.r2_score(y_binary, y)

# AUC manual simplificado para BigModel
thresholds_auc = np.linspace(0, 1, 100)
tpr_l, fpr_l = [], []
for t in thresholds_auc:
    y_t = (y >= t).astype(int)
    tp = np.sum((y_t == 1) & (y_binary == 1))
    fn = np.sum((y_t == 0) & (y_binary == 1))
    fp = np.sum((y_t == 1) & (y_binary == 0))
    tn = np.sum((y_t == 0) & (y_binary == 0))
    tpr_l.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fpr_l.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

if hasattr(np, 'trapezoid'):
    auc_big = np.trapezoid(tpr_l[::-1], fpr_l[::-1])
else:
    auc_big = np.trapz(tpr_l[::-1], fpr_l[::-1])

print(f"Calidad del BigModel (vs Ground Truth): R2 = {r2_big:.4f} | AUC = {auc_big:.4f}")


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

# --- Graficar evolución del aprendizaje ---
plots.graficar_mse_epochs(hist_lin, hist_nolin, os.path.join(os.path.dirname(__file__), 'plot_mse_epochs.png'))

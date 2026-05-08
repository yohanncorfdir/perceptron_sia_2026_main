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

# Division train/test (80 % / 20 %)
np.random.seed(42)
idx_perm = np.random.permutation(len(X))
split = int(0.8 * len(X))
idx_train, idx_test = idx_perm[:split], idx_perm[split:]
X_train, y_train = X[idx_train], y[idx_train]
X_test,  y_test  = X[idx_test],  y[idx_test]

EPOCHS = 30
print(f"Conjunto de datos: {len(X)} muestras, {X.shape[1]} variables de entrada")
print(f"Particion: {len(X_train)} muestras en train, {len(X_test)} muestras en test")
print(f"Target: Knowledge Distillation (replicar BigModel)")
print(f"Media de probabilidad de BigModel: {y.mean():.4f}")

# Evaluación de la calidad del BigModel 
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


# Estudio del potencial de aprendizaje (train 80 %, evaluacion en test 20 %)
pl  = PerceptronLineal(X_train.shape[1], alpha=0.1)
pnl = PerceptronNoLineal(X_train.shape[1], alpha=0.1)

mse_lin_train,   mse_lin_test   = pl.fit(X_train,  y_train, epochs=EPOCHS, X_val=X_test,  y_val=y_test)
mse_nolin_train, mse_nolin_test = pnl.fit(X_train, y_train, epochs=EPOCHS, X_val=X_test,  y_val=y_test)

# Conversion MSE -> RMSE
rmse_lin_train   = np.sqrt(mse_lin_train)
rmse_lin_test    = np.sqrt(mse_lin_test)
rmse_nolin_train = np.sqrt(mse_nolin_train)
rmse_nolin_test  = np.sqrt(mse_nolin_test)

print(f"\n{'Época':<8} {'Lineal RMSE train':<22} {'Lineal RMSE test':<22} {'No Lineal RMSE train':<24} {'No Lineal RMSE test':<22}")

for i, (rt_l, rte_l, rt_nl, rte_nl) in enumerate(
        zip(rmse_lin_train, rmse_lin_test, rmse_nolin_train, rmse_nolin_test), 1):
    print(f"{i:<8} {rt_l:<22.6f} {rte_l:<22.6f} {rt_nl:<24.6f} {rte_nl:<22.6f}")

print(f"\nRMSE Final Lineal    — train: {rmse_lin_train[-1]:.6f}  |  test: {rmse_lin_test[-1]:.6f}")
print(f"RMSE Final No Lineal — train: {rmse_nolin_train[-1]:.6f}  |  test: {rmse_nolin_test[-1]:.6f}")

#Analisis de underfitting y saturacion

# Underfitting: RMSE alto incluso sobre los datos de entrenamiento
umbral_underfitting_rmse = np.sqrt(0.05)  # ~0.224
for nombre, hist in [("Lineal", rmse_lin_train), ("No Lineal", rmse_nolin_train)]:
    if min(hist) > umbral_underfitting_rmse:
        print(f"{nombre}: UNDERFITTING detectado (RMSE minimo train = {min(hist):.6f})")
    else:
        print(f"{nombre}: Sin underfitting (RMSE minimo train = {min(hist):.6f})")

# Saturacion: el RMSE de train deja de mejorar
for nombre, hist in [("Lineal", rmse_lin_train), ("No Lineal", rmse_nolin_train)]:
    ultimas = hist[-5:]
    rango_final = max(ultimas) - min(ultimas)
    if rango_final < 0.001:
        print(f"{nombre}: SATURACION detectada (variacion en ultimas 5 epocas = {rango_final:.6f})")
    else:
        print(f"{nombre}: Sin saturacion (progresion RMSE train {hist[0]:.6f} -> {hist[-1]:.6f})")

# Validacion Cruzada K-Fold (K=5) para seleccion de modelo 
K = 5
print(f"\n{'='*70}")
print(f"Validacion Cruzada Estratificada K-Fold (K={K}) — Comparacion de modelos")
print(f"{'='*70}")

# Indices estratificados por clase real para mantener la proporcion de fraudes
idx_fraude    = np.where(y_binary == 1)[0]
idx_nofraude  = np.where(y_binary == 0)[0]
np.random.seed(42)
np.random.shuffle(idx_fraude)
np.random.shuffle(idx_nofraude)

folds_fraude   = np.array_split(idx_fraude,   K)
folds_nofraude = np.array_split(idx_nofraude, K)
folds = [np.concatenate([folds_fraude[k], folds_nofraude[k]]) for k in range(K)]

rmse_cv_lin  = []
rmse_cv_nolin = []

print(f"\n{'Fold':<6} {'RMSE Lineal':<18} {'RMSE No Lineal':<18}")
print("-" * 42)

for k in range(K):
    idx_val   = folds[k]
    idx_cv_tr = np.concatenate([folds[j] for j in range(K) if j != k])

    X_cv_train, y_cv_train = X[idx_cv_tr], y[idx_cv_tr]
    X_cv_val,   y_cv_val   = X[idx_val],   y[idx_val]

    m_lin  = PerceptronLineal(X_cv_train.shape[1],  alpha=0.1)
    m_nlin = PerceptronNoLineal(X_cv_train.shape[1], alpha=0.1)

    mse_tr_l,  mse_val_l  = m_lin.fit(X_cv_train,  y_cv_train, epochs=EPOCHS,
                                       X_val=X_cv_val, y_val=y_cv_val)
    mse_tr_nl, mse_val_nl = m_nlin.fit(X_cv_train, y_cv_train, epochs=EPOCHS,
                                        X_val=X_cv_val, y_val=y_cv_val)

    rmse_l  = np.sqrt(mse_val_l[-1])
    rmse_nl = np.sqrt(mse_val_nl[-1])
    rmse_cv_lin.append(rmse_l)
    rmse_cv_nolin.append(rmse_nl)

    print(f"{k+1:<6} {rmse_l:<18.6f} {rmse_nl:<18.6f}")

print("-" * 42)
media_lin  = np.mean(rmse_cv_lin)
media_nlin = np.mean(rmse_cv_nolin)
std_lin    = np.std(rmse_cv_lin)
std_nlin   = np.std(rmse_cv_nolin)
print(f"{'Media':<6} {media_lin:<18.6f} {media_nlin:<18.6f}")
print(f"{'Std':<6} {std_lin:<18.6f} {std_nlin:<18.6f}")

# Seleccion del modelo segun RMSE medio en validacion cruzada
mejor = "No Lineal" if media_nlin <= media_lin else "Lineal"
print(f"\nModelo seleccionado por validacion cruzada: {mejor}")
print(f"  Lineal    -> RMSE medio = {media_lin:.6f} ± {std_lin:.6f}")
print(f"  No Lineal -> RMSE medio = {media_nlin:.6f} ± {std_nlin:.6f}")

# --- Umbral por acuerdo máximo con BigModel (Knowledge Distillation) ---
y_prob_test  = pnl.predict_proba(X_test).flatten()
y_big_binary = (y_test >= 0.5).astype(int)

umbrales      = np.linspace(0.01, 0.99, 99)
acuerdos      = [np.mean((y_prob_test >= t).astype(int) == y_big_binary) for t in umbrales]
umbral_optimo = umbrales[np.argmax(acuerdos)]

print(f"\n{'='*50}")
print(f"Umbral óptimo por acuerdo con BigModel: {umbral_optimo:.2f}")
print(f"Acuerdo máximo sobre test:              {max(acuerdos)*100:.1f} %")
print(f"(referencia: umbral 0.5 → acuerdo = {np.mean((y_prob_test >= 0.5).astype(int) == y_big_binary)*100:.1f} %)")

plots.graficar_acuerdo_umbral(umbrales, acuerdos, umbral_optimo,
                              os.path.join(os.path.dirname(__file__), 'plot_acuerdo_umbral.png'))

y_tiny_binary = (y_prob_test >= umbral_optimo).astype(int)
plots.graficar_confusion_kd(y_big_binary, y_tiny_binary, umbral_optimo, max(acuerdos),
                            os.path.join(os.path.dirname(__file__), 'plot_confusion_kd.png'))

#Graficar evolución del RMSE (train y test)
plots.graficar_rmse_epochs(rmse_lin_train, rmse_lin_test,
                           rmse_nolin_train, rmse_nolin_test,
                           os.path.join(os.path.dirname(__file__), 'plot_rmse_epochs.png'))

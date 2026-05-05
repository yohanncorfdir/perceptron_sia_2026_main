import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptronSimpleNoLineal import PerceptronNoLineal
import plots


# Funciones de metricas 
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn

def metricas(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1


# Carga y normalizacion
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/fraud_dataset.csv'))

# Guardamos amount_usd para el análisis económico
amount_usd = df['amount_usd'].values.astype(float)

# Usamos big_model_fraud_probability como target para Knowledge Distillation
X = df.drop(columns=['flagged_fraud', 'big_model_fraud_probability']).values.astype(float)
y = df['big_model_fraud_probability'].values.astype(float)
y_binary = df['flagged_fraud'].values.astype(int)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

# Calidad del BigModel (K-Fold completo)
r2_big = plots.r2_score(y_binary, y)
print(f"Calidad del BigModel (General): R2 = {r2_big:.4f}")
EPOCHS = 30
K      = 5

print(f"Modelo: PerceptronNoLineal | Epochs: {EPOCHS} | K-Fold: {K}")
print(f"Muestras: {len(X)} | Target: Knowledge Distillation")
print(f"Fraudes Reales: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")


# Stratified K-Fold Cross Validation
# Mantenemos la estratificacion basada en y_binary para asegurar folds balanceados
print("Stratified K-Fold Cross Validation (K=5)")

# Construimos los indices estratificados manualmente basados en etiquetas reales
idx_fraud    = np.where(y_binary == 1)[0]
idx_nofraud  = np.where(y_binary == 0)[0]
np.random.seed(42)
np.random.shuffle(idx_fraud)
np.random.shuffle(idx_nofraud)

folds_fraud   = np.array_split(idx_fraud,   K)
folds_nofraud = np.array_split(idx_nofraud, K)
folds = [np.concatenate([folds_fraud[k], folds_nofraud[k]]) for k in range(K)]

resultados = []

print(f"\n{'Fold':<6} {'MSE':<10} {'Acc %':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
print("-" * 62)

for k in range(K):
    idx_test  = folds[k]
    idx_train = np.concatenate([folds[i] for i in range(K) if i != k])

    X_train, y_train = X[idx_train], y[idx_train]
    X_test,  y_test  = X[idx_test],  y[idx_test]
    y_test_binary    = y_binary[idx_test]

    modelo = PerceptronNoLineal(X_train.shape[1], alpha=0.1)
    modelo.fit(X_train, y_train, epochs=EPOCHS)

    # Prediccion continua para MSE
    y_prob = modelo.predict_proba(X_test).flatten()
    mse = np.mean((y_prob - y_test)**2)
    
    # Prediccion binaria para metricas de clasificacion contra ground truth real
    y_pred = modelo.predict(X_test, threshold=0.5)
    acc, prec, rec, f1 = metricas(y_test_binary, y_pred)
    
    resultados.append((mse, acc, prec, rec, f1))
    print(f"{k+1:<6} {mse:<10.4f} {acc*100:<10.2f} {prec:<12.3f} {rec:<10.3f} {f1:<10.3f}")

resultados = np.array(resultados)
print("-" * 62)
print(f"{'Media':<6} {resultados[:,0].mean():<10.4f} "
      f"{resultados[:,1].mean()*100:<10.2f} "
      f"{resultados[:,2].mean():<12.3f} "
      f"{resultados[:,3].mean():<10.3f} "
      f"{resultados[:,4].mean():<10.3f}")
print(f"{'Std':<6} {resultados[:,0].std():<10.4f} "
      f"{resultados[:,1].std()*100:<10.2f} "
      f"{resultados[:,2].std():<12.3f} "
      f"{resultados[:,3].std():<10.3f} "
      f"{resultados[:,4].std():<10.3f}")


# MEJOR MODELO 
# Entrenamos el modelo final sobre el 80% y evaluamos el umbral en el 20%
split    = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:] # Target continuo para train
y_test_real = y_binary[split:]         # Ground truth para evaluacion

modelo_final = PerceptronNoLineal(X_train.shape[1], alpha=0.1)
modelo_final.fit(X_train, y_train, epochs=EPOCHS)

# --- Análisis Económico ---
# Costo de oportunidad: 10% del monto si bloqueamos por error (FP)
# Costo de pérdida directa: 100% del monto si omitimos un fraude (FN)
FACTOR_FP = 0.1 

amounts_test = amount_usd[split:]
y_test_proba = modelo_final.predict_proba(X_test).flatten()

thresholds = np.linspace(0.01, 0.99, 50)
precisiones, recalls, f1s = [], [], []
total_costs, fn_costs, fp_costs = [], [], []

for t in thresholds:
    y_pred = (y_test_proba >= t).astype(int)
    
    # Métricas estándar
    _, prec, rec, f1 = metricas(y_test_real, y_pred)
    precisiones.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    
    # Cálculo económico
    fp_idx = (y_pred == 1) & (y_test_real == 0)
    fn_idx = (y_pred == 0) & (y_test_real == 1)
    
    cost_fp = np.sum(amounts_test[fp_idx]) * FACTOR_FP
    cost_fn = np.sum(amounts_test[fn_idx])
    
    fp_costs.append(cost_fp)
    fn_costs.append(cost_fn)
    total_costs.append(cost_fp + cost_fn)

mejor_idx_f1 = np.argmax(f1s)
mejor_idx_eco = np.argmin(total_costs)

umbral_f1  = thresholds[mejor_idx_f1]
umbral_eco = thresholds[mejor_idx_eco]

print(f"\n{'Umbral':<10} {'F1-Score':<10} {'Pérdida USD':<15}")
print("-" * 40)
for i in [0, 10, 20, 30, 40, 49]: # Muestra representativa
    t, f, c = thresholds[i], f1s[i], total_costs[i]
    print(f"{t:<10.2f} {f:<10.3f} ${c:<14,.0f}")

print(f"\nResultados del Análisis:")
print(f"  -> Umbral óptimo por F1: {umbral_f1:.2f} (Pérdida: ${total_costs[mejor_idx_f1]:,.0f})")
print(f"  -> Umbral óptimo ECONÓMICO: {umbral_eco:.2f} (Pérdida: ${total_costs[mejor_idx_eco]:,.0f})")

# Graficamos el impacto económico
plots.graficar_costo_economico(thresholds, total_costs, fn_costs, fp_costs, umbral_eco, 
                               os.path.join(os.path.dirname(__file__), 'plot_costo_economico.png'))

# Graficamos métricas estándar vs umbral
plots.graficar_umbral(thresholds, precisiones, recalls, f1s, umbral_f1, 
                      os.path.join(os.path.dirname(__file__), 'plot_umbral.png'))

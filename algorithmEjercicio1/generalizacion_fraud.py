import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptronSimpleNoLineal import PerceptronNoLineal


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

# Usamos big_model_fraud_probability como target para Knowledge Distillation
X = df.drop(columns=['flagged_fraud', 'big_model_fraud_probability']).values.astype(float)
y = df['big_model_fraud_probability'].values.astype(float)
y_binary = df['flagged_fraud'].values.astype(int)

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

EPOCHS = 20
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

# Barrido de umbrales
thresholds  = np.arange(0.1, 0.91, 0.05)
precisiones, recalls, f1s = [], [], []

for t in thresholds:
    y_pred = modelo_final.predict(X_test, threshold=t)
    _, prec, rec, f1 = metricas(y_test_real, y_pred)
    precisiones.append(prec)
    recalls.append(rec)
    f1s.append(f1)

mejor_idx    = np.argmax(f1s)
mejor_umbral = thresholds[mejor_idx]

print(f"\n{'Umbral':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")

for t, p, r, f in zip(thresholds, precisiones, recalls, f1s):
    marca = " <-- OPTIMO (F1)" if abs(t - mejor_umbral) < 0.001 else ""
    print(f"{t:<10.2f} {p:<12.3f} {r:<10.3f} {f:<10.3f}{marca}")

print(f"\nUmbral recomendado para CompanyX: {mejor_umbral:.2f}")
print(f"  -> F1={f1s[mejor_idx]:.3f} | Precision={precisiones[mejor_idx]:.3f} | Recall={recalls[mejor_idx]:.3f}")


# --- Grafico Precision / Recall / F1 vs Umbral ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisiones, 'o-', color='steelblue',  label='Precision')
ax.plot(thresholds, recalls,     's-', color='crimson',    label='Recall')
ax.plot(thresholds, f1s,         '^-', color='seagreen',   label='F1-Score')
ax.axvline(mejor_umbral, color='gray', linestyle='--', alpha=0.7,
           label=f'Umbral optimo = {mejor_umbral:.2f}')
ax.set_xlabel('Umbral de deteccion', fontsize=11)
ax.set_ylabel('Valor de la metrica', fontsize=11)
ax.set_title('Precision / Recall / F1 segun el umbral de deteccion de fraude', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_umbral.png'), dpi=150)
# plt.show()  <-- Comentado para evitar warnings en entornos no interactivos
print("\nGrafico guardado: plot_umbral.png")

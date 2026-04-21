import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlp_v2 import MLPv2

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

# =============================================================
# Carga de datos
# =============================================================
def cargar(path):
    df = pd.read_csv(path)
    X = np.array([np.array(ast.literal_eval(s), dtype=np.float32) for s in df['image']])
    y = df['label'].values.astype(int)
    return X, y

X_base,  y_base  = cargar(os.path.join(DATA_DIR, 'digits.csv'))
X_more,  y_more  = cargar(os.path.join(DATA_DIR, 'more_digits.csv'))
X_test,  y_test  = cargar(os.path.join(DATA_DIR, 'digits_test.csv'))

# c) Factor clave: combinacion de los dos conjuntos de entrenamiento
X_train = np.concatenate([X_base, X_more], axis=0)
y_train = np.concatenate([y_base, y_more], axis=0)

print(f"Train solo digits.csv   : {X_base.shape[0]} muestras")
print(f"Train more_digits.csv   : {X_more.shape[0]} muestras")
print(f"Train combinado         : {X_train.shape[0]} muestras")
print(f"Test                    : {X_test.shape[0]} muestras")

# =============================================================
# b) Tecnica 1: Data Augmentation
#    Genera imagenes desplazadas para enriquecer el conjunto
# =============================================================
def augment(X, y, shifts=[(0,1),(0,-1),(1,0),(-1,0)]):
    """Desplazamiento de 1 pixel en 4 direcciones -> 4x mas datos."""
    X_aug, y_aug = [X], [y]
    for dr, dc in shifts:
        imgs = X.reshape(-1, 28, 28)
        imgs_shifted = np.roll(imgs, shift=dr, axis=1)
        imgs_shifted = np.roll(imgs_shifted, shift=dc, axis=2)
        X_aug.append(imgs_shifted.reshape(-1, 784))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)

print("\nAplicando data augmentation (desplazamientos)...")
X_aug, y_aug = augment(X_train, y_train)
print(f"Train tras augmentation : {X_aug.shape[0]} muestras")

# =============================================================
# Metricas
# =============================================================
def reporte(y_true, y_pred, nombre=""):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    acc = np.trace(cm) / cm.sum()
    f1s = []
    for c in range(10):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)
    print(f"\n{'='*55}")
    print(f"Metricas: {nombre}")
    print(f"Exactitud: {acc*100:.2f}%  |  Macro-F1: {np.mean(f1s):.3f}")
    return acc, cm

# =============================================================
# b) Variantes para alcanzar 98%
# =============================================================
VARIANTES = [
    # Referencia: mejor modelo del ejercicio 2 (sigmoid, sin augmentation)
    {"nombre": "Ref: sigmoid [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "sigmoid",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    # b.1) Solo cambiar activacion a ReLU
    {"nombre": "ReLU      [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    # b.2) ReLU + arquitectura mas grande
    {"nombre": "ReLU      [512,256] adam  sin aug",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    # b.3) ReLU + augmentation
    {"nombre": "ReLU      [256,128] adam  CON aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_aug, "y": y_aug, "epochs": 80},

    # b.4) ReLU + arquitectura grande + augmentation + lr_decay
    {"nombre": "ReLU      [512,256] adam  CON aug + decay",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.005, "X": X_aug, "y": y_aug, "epochs": 80},
]

resultados = []
for v in VARIANTES:
    print(f"\n>>> {v['nombre']}")
    np.random.seed(42)
    modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                   alpha=v['alpha'], optimizador=v['opt'],
                   activacion=v['act'], lr_decay=v['decay'])
    hist_tr, hist_val = modelo.fit(
        v['X'], v['y'],
        epochs=v['epochs'], batch_size=64,
        X_val=X_test, y_val=y_test,
        verbose=True
    )
    acc_test = modelo.score(X_test, y_test)
    resultados.append({
        "nombre": v['nombre'], "modelo": modelo,
        "hist_tr": hist_tr, "hist_val": hist_val,
        "acc_test": acc_test,
    })

# =============================================================
# a) Mejor resultado
# =============================================================
print("\n" + "="*65)
print("a) Resumen - mejores resultados obtenidos")
print("="*65)
print(f"{'Variante':<45} {'Acc test %'}")
print("-"*58)
for r in resultados:
    marca = " <-- MEJOR" if r == max(resultados, key=lambda x: x['acc_test']) else ""
    print(f"{r['nombre']:<45} {r['acc_test']*100:.2f}%{marca}")

mejor = max(resultados, key=lambda r: r['acc_test'])
acc_final, cm = reporte(y_test, mejor['modelo'].predict(X_test), nombre=mejor['nombre'])

objetivo = acc_final >= 0.98
print(f"\nObjetivo 98% {'ALCANZADO' if objetivo else 'NO alcanzado aun'}: {acc_final*100:.2f}%")

# =============================================================
# c) Factores externos
# =============================================================
print("\n" + "="*65)
print("c) Factores que influyeron en el cambio de rendimiento")
print("="*65)
print(f"  Train solo digits.csv  : {X_base.shape[0]} muestras")
print(f"  Train combinado + aug  : {X_aug.shape[0]} muestras ({X_aug.shape[0]/X_base.shape[0]:.1f}x mas)")
print("  -> Mas datos = mejor generalizacion (menos overfitting)")
print("  -> Datos mas diversos = el modelo ve mas variantes de cada digito")

# =============================================================
# Graficos
# =============================================================
fig, ax = plt.subplots(figsize=(11, 5))
colores = plt.cm.tab10(np.linspace(0, 1, len(resultados)))
for r, color in zip(resultados, colores):
    ax.plot(r['hist_val'], label=f"{r['nombre']} ({r['acc_test']*100:.1f}%)",
            color=color, linewidth=1.5)
ax.axhline(0.98, color='black', linestyle='--', linewidth=1, label='Objetivo 98%')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Exactitud en test', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_title('Ejercicio 3 - Curvas de aprendizaje hacia 98%', fontsize=12)
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_curvas_98.png'), dpi=150)
plt.show()

# Matriz de confusion del mejor modelo
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel('Prediccion', fontsize=12); ax.set_ylabel('Real', fontsize=12)
ax.set_title(f'Matriz de confusion - {mejor["nombre"]}', fontsize=10)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_confusion_98.png'), dpi=150)
plt.show()

print("\nGraficos: plot_curvas_98.png | plot_confusion_98.png")

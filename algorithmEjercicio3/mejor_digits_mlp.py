import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlp_v2 import MLPv2

DATA_DIR    = os.path.join(os.path.dirname(__file__), '../data')
MODELOS_DIR = os.path.join(os.path.dirname(__file__), 'modelos')
os.makedirs(MODELOS_DIR, exist_ok=True)


# Carga de datos
def cargar(path):
    df = pd.read_csv(path)
    X = np.array([np.array(ast.literal_eval(s), dtype=np.float32) for s in df['image']])
    y = df['label'].values.astype(int)
    return X, y

X_base,  y_base  = cargar(os.path.join(DATA_DIR, 'digits.csv'))
X_more,  y_more  = cargar(os.path.join(DATA_DIR, 'more_digits.csv'))
X_test,  y_test  = cargar(os.path.join(DATA_DIR, 'digits_test.csv'))

#Combinacion de los dos conjuntos de entrenamiento
X_train = np.concatenate([X_base, X_more], axis=0)
y_train = np.concatenate([y_base, y_more], axis=0)


print(f"Train: {X_train.shape[0]} muestras")
print(f"Test : {X_test.shape[0]}  muestras")


# Metricas para evaluar nuestros modelos
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
    print(f"Metricas: {nombre}")
    print(f"Exactitud: {acc*100:.2f}%  |  Macro-F1: {np.mean(f1s):.3f}")
    return acc, cm


# Genera un nombre de archivo unico a partir del nombre de la variante
def _ruta_modelo(nombre):
    safe = nombre.strip().replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
    return os.path.join(MODELOS_DIR, safe + ".npz")


#  Variantes
VARIANTES = [
    # activacion sigmoid
    {"nombre": "Ref: sigmoid [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "sigmoid",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    # activacion a ReLU
    {"nombre": "ReLU      [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    # ReLU + arquitectura mas grande
    {"nombre": "ReLU      [512,256] adam  sin aug",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},
]

resultados = []
for v in VARIANTES:
    print(f"\n>>> {v['nombre']}")
    ruta = _ruta_modelo(v['nombre'])

    if os.path.exists(ruta):
        # Modelo ya entrenado: lo cargamos directamente sin reentrenar
        print(f"    [cargando desde {ruta}]")
        modelo    = MLPv2.cargar(ruta)
        hist_tr   = []
        hist_val  = []
    else:
        np.random.seed(42)
        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])
        hist_tr, hist_val = modelo.fit(
            v['X'], v['y'],
            epochs=v['epochs'], batch_size=64,
            X_val=X_test, y_val=y_test,
            verbose=True,
            paciencia=10, min_delta=0.001
        )
        modelo.guardar(ruta)
        print(f"    [modelo guardado en {ruta}]")

    acc_test = modelo.score(X_test, y_test)
    resultados.append({
        "nombre": v['nombre'], "modelo": modelo,
        "hist_tr": hist_tr, "hist_val": hist_val,
        "acc_test": acc_test,
    })


# Mejor resultado

print("Resumen ")
print(f"{'Variante':<45} {'Acc test %'}")

for r in resultados:
    marca = " <-- MEJOR" if r == max(resultados, key=lambda x: x['acc_test']) else ""
    print(f"{r['nombre']:<45} {r['acc_test']*100:.2f}%{marca}")

mejor = max(resultados, key=lambda r: r['acc_test'])
acc_final, cm = reporte(y_test, mejor['modelo'].predict(X_test), nombre=mejor['nombre'])


# Graficos
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


#  Tecnica para mejorar: Data Augmentation
#  Genera imagenes desplazadas para enriquecer el conjunto

def augment(X, y, shifts=[(0,1),(0,-1),(1,0),(-1,0)]):
    #Desplazamiento de 1 pixel en 4 direcciones -> 4x mas datos
    X_aug, y_aug = [X], [y]
    for dr, dc in shifts:
        imgs = X.reshape(-1, 28, 28)
        imgs_shifted = np.roll(imgs, shift=dr, axis=1)
        imgs_shifted = np.roll(imgs_shifted, shift=dc, axis=2)
        X_aug.append(imgs_shifted.reshape(-1, 784))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)
X_aug, y_aug = augment(X_train, y_train)


# Variantes con augmentacion
VARIANTES_Aug = [

    # ReLU + augmentation
    {"nombre": "ReLU      [256,128] adam  CON aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_aug, "y": y_aug, "epochs": 80},

    # ReLU + arquitectura grande + augmentation + lr_decay
    {"nombre": "ReLU      [512,256] adam  CON aug + decay",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.005, "X": X_aug, "y": y_aug, "epochs": 80},
]

resultados = []
for v in VARIANTES_Aug:
    print(f"\n>>> {v['nombre']}")
    ruta = _ruta_modelo("aug_" + v['nombre'])

    if os.path.exists(ruta):
        # Modelo ya entrenado con augmentation: lo cargamos directamente
        print(f"    [cargando desde {ruta}]")
        modelo   = MLPv2.cargar(ruta)
        hist_tr  = []
        hist_val = []
    else:
        np.random.seed(42)
        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])
        hist_tr, hist_val = modelo.fit(
            v['X'], v['y'],
            epochs=v['epochs'], batch_size=64,
            X_val=X_test, y_val=y_test,
            verbose=True,
            paciencia=10, min_delta=0.001
        )
        modelo.guardar(ruta)
        print(f"    [modelo guardado en {ruta}]")

    acc_test = modelo.score(X_test, y_test)
    resultados.append({
        "nombre": v['nombre'], "modelo": modelo,
        "hist_tr": hist_tr, "hist_val": hist_val,
        "acc_test": acc_test,
    })


# Cross validation para generalizar mas el modelo

K = 5

print(f"Stratified K-Fold Cross Validation | K={K} | Muestras aug: {len(X_aug)}")

# Construimos los indices estratificados manualmente, uno por clase
idx_0 = np.where(y_aug == 0)[0]
idx_1 = np.where(y_aug == 1)[0]
idx_2 = np.where(y_aug == 2)[0]
idx_3 = np.where(y_aug == 3)[0]
idx_4 = np.where(y_aug == 4)[0]
idx_5 = np.where(y_aug == 5)[0]
idx_6 = np.where(y_aug == 6)[0]
idx_7 = np.where(y_aug == 7)[0]
idx_8 = np.where(y_aug == 8)[0]
idx_9 = np.where(y_aug == 9)[0]

# Mezclamos aleatoriamente dentro de cada clase antes de dividir
np.random.seed(42)
np.random.shuffle(idx_0)
np.random.shuffle(idx_1)
np.random.shuffle(idx_2)
np.random.shuffle(idx_3)
np.random.shuffle(idx_4)
np.random.shuffle(idx_5)
np.random.shuffle(idx_6)
np.random.shuffle(idx_7)
np.random.shuffle(idx_8)
np.random.shuffle(idx_9)

# Dividimos cada clase en K bloques iguales
folds_idx0 = np.array_split(idx_0, K)
folds_idx1 = np.array_split(idx_1, K)
folds_idx2 = np.array_split(idx_2, K)
folds_idx3 = np.array_split(idx_3, K)
folds_idx4 = np.array_split(idx_4, K)
folds_idx5 = np.array_split(idx_5, K)
folds_idx6 = np.array_split(idx_6, K)
folds_idx7 = np.array_split(idx_7, K)
folds_idx8 = np.array_split(idx_8, K)
folds_idx9 = np.array_split(idx_9, K)

# Combinamos un bloque de cada clase para formar cada fold estratificado
folds = [np.concatenate([folds_idx0[k], folds_idx1[k], folds_idx2[k], folds_idx3[k],
                          folds_idx4[k], folds_idx5[k], folds_idx6[k], folds_idx7[k],
                          folds_idx8[k], folds_idx9[k]]) for k in range(K)]


# Variantes a evaluar con cross validation
VARIANTES_CrossVal = [

    # ReLU + augmentation
    {"nombre": "ReLU      [256,128] adam  CON aug + CV",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "epochs": 80},

    # ReLU + arquitectura grande + augmentation + lr_decay
    {"nombre": "ReLU      [512,256] adam  CON aug + CV + decay",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.005, "epochs": 80},
]

resultados_cv = []

print(f"\n{'Variante':<45} {'Fold':<6} {'Acc %':<10} {'Macro-F1':<10}")
print("-" * 75)
for v in VARIANTES_CrossVal:
    accs_fold = []
    for k in range(K):
        # Separamos indices de test (fold k) y train (los demas folds)
        idx_test_cv  = folds[k]
        idx_train_cv = np.concatenate([folds[i] for i in range(K) if i != k])

        # Extraemos los datos del fold correspondiente desde X_aug/y_aug
        X_train_cv, y_train_cv = X_aug[idx_train_cv], y_aug[idx_train_cv]
        X_test_cv,  y_test_cv  = X_aug[idx_test_cv],  y_aug[idx_test_cv]

        # Entrenamos el modelo solo con los datos de entrenamiento del fold
        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])

        hist_tr, hist_val = modelo.fit(
            X_train_cv, y_train_cv,
            epochs=v['epochs'], batch_size=64,
            X_val=X_test_cv, y_val=y_test_cv,
            verbose=False,
            paciencia=10, min_delta=0.001
        )

        # Calculamos exactitud y Macro-F1 sobre el fold de test
        acc_fold  = modelo.score(X_test_cv, y_test_cv)
        y_pred_cv = modelo.predict(X_test_cv)

        f1s = []
        for c in range(10):
            tp = np.sum((y_pred_cv == c) & (y_test_cv == c))
            fp = np.sum((y_pred_cv == c) & (y_test_cv != c))
            fn = np.sum((y_pred_cv != c) & (y_test_cv == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        macro_f1 = np.mean(f1s)

        accs_fold.append(acc_fold)
        print(f"{v['nombre']:<45} {k+1:<6} {acc_fold*100:<10.2f} {macro_f1:<10.3f}")
        resultados_cv.append({
            "nombre": v['nombre'], "fold": k,
            "acc_test": acc_fold, "macro_f1": macro_f1,
            "hist_tr": hist_tr, "hist_val": hist_val,
        })

    # Resumen por variante: media y desviacion estandar de la exactitud entre folds
    print(f"  --> Media: {np.mean(accs_fold)*100:.2f}%  |  Std: {np.std(accs_fold)*100:.2f}%\n")
        


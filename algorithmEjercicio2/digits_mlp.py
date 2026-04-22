import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlp import MLP

# Importar los datos
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def cargar(path):
    df = pd.read_csv(path)
    X = np.array([np.array(ast.literal_eval(s), dtype=np.float32)
                  for s in df['image']])
    y = df['label'].values.astype(int)
    return X, y

X_train, y_train = cargar(os.path.join(DATA_DIR, 'digits.csv'))
X_test,  y_test  = cargar(os.path.join(DATA_DIR, 'digits_test.csv'))

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Clases: {np.unique(y_train)}")


# Metricas de evaluacion
def matriz_confusion(y_true, y_pred, n_clases=10):
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def reporte_metricas(y_true, y_pred, n_clases=10): #Calculo de las metricas en funcion de y_pred y y_true
    cm = matriz_confusion(y_true, y_pred, n_clases)
    acc = np.trace(cm) / cm.sum()
    precisiones, recalls, f1s = [], [], []
    for c in range(n_clases):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        precisiones.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return acc, precisiones, recalls, f1s, cm

def imprimir_reporte(acc, precisiones, recalls, f1s, nombre=""):
    print(f"\n{'='*55}")
    print(f"Metricas: {nombre}")
    print(f"{'='*55}")
    print(f"Exactitud global: {acc*100:.2f}%")
    print(f"\n{'Digit':<8} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 42)
    for c in range(10):
        print(f"  {c:<6} {precisiones[c]:<12.3f} {recalls[c]:<10.3f} {f1s[c]:<10.3f}")
    print(f"\nMacro-F1: {np.mean(f1s):.3f}")


#Probamos distintas arquitectura 

VARIANTES = [
    {"capas": [32,16],       "alpha": 0.1,  "epochs": 50}, #Una capa de 32 y una de 16
    {"capas": [64,32],       "alpha": 0.1,  "epochs": 50}, #Una capa de 64 y una de 32
    {"capas": [64],       "alpha": 0.1,  "epochs": 50}, #Una capa de 64
    {"capas": [128],      "alpha": 0.1,  "epochs": 50}, #Una capa de 128
    {"capas": [128, 64],  "alpha": 0.1, "epochs": 50}, #Una capa de 128 y de 64
    {"capas": [256, 128], "alpha": 0.1, "epochs": 50}, #Una capa de 256 y de 128
    {"capas": [32,16],       "alpha": 0.05,  "epochs": 50}, #Una capa de 32 y una de 16
    {"capas": [64,32],       "alpha": 0.05,  "epochs": 50}, #Una capa de 64 y una de 32
    {"capas": [64],       "alpha": 0.05,  "epochs": 50}, #Una capa de 64
    {"capas": [128],      "alpha": 0.05,  "epochs": 50}, #Una capa de 128
    {"capas": [128, 64],  "alpha": 0.05, "epochs": 50}, #Una capa de 128 y de 64
    {"capas": [256, 128], "alpha": 0.05, "epochs": 50}, #Una capa de 256 y de 128
]


# Variantes de arquitectura + tasa de aprendizaje (SGD)
resultados = []
for v in VARIANTES:
    nombre = f"sgd  ocultas={v['capas']}  alpha={v['alpha']}"
    print(f"\n>>> {nombre}")
    np.random.seed(42)
    modelo = MLP(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                 alpha=v['alpha'], optimizador='sgd')
    hist_tr, hist_val = modelo.fit(
        X_train, y_train,
        epochs=v['epochs'], batch_size=32,
        X_val=X_test, y_val=y_test,
        verbose=True
    )
    resultados.append({
        "nombre": nombre,
        "modelo": modelo,
        "hist_tr": hist_tr,
        "hist_val": hist_val,
        "acc_test": modelo.score(X_test, y_test),
        "grupo": "SGD",
    })

VARIANTES_OPT = [
    {"capas": [128, 64], "alpha": 0.1,   "opt": "sgd",      "epochs": 50},
    {"capas": [128, 64], "alpha": 0.1,   "opt": "momentum", "epochs": 50},
    {"capas": [128, 64], "alpha": 0.01,  "opt": "adam",     "epochs": 50},
    {"capas": [256, 128],"alpha": 0.1,   "opt": "momentum", "epochs": 50},
    {"capas": [256, 128],"alpha": 0.01,  "opt": "adam",     "epochs": 50},
]

#Parece que momentu tiene overfitting muy rapido. 

#Variantes de mecanismo de optimizacion
#SGD     : descenso por gradiente clasico
#Momentum: acumula velocidad -> converge mas rapido
#Adam    : adapta alpha por peso -> mas estable

for v in VARIANTES_OPT:
    nombre = f"{v['opt']:<10} ocultas={v['capas']}  alpha={v['alpha']}"
    print(f"\n>>> {nombre}")
    np.random.seed(42)
    modelo = MLP(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                 alpha=v['alpha'], optimizador=v['opt'])
    hist_tr, hist_val = modelo.fit(
        X_train, y_train,
        epochs=v['epochs'], batch_size=32,
        X_val=X_test, y_val=y_test,
        verbose=True
    )
    resultados.append({
        "nombre": nombre,
        "modelo": modelo,
        "hist_tr": hist_tr,
        "hist_val": hist_val,
        "acc_test": modelo.score(X_test, y_test),
        "grupo": v['opt'].upper(),
    })


# Resumen comparativo global
print("Resumen comparativo de todas las variantes")
print(f"{'Variante':<50} {'Acc test %'}")

for r in resultados:
    print(f"{r['nombre']:<50} {r['acc_test']*100:.2f}%")

mejor = max(resultados, key=lambda r: r['acc_test'])
print(f"\nMejor modelo: {mejor['nombre']}  ->  {mejor['acc_test']*100:.2f}%")


# Curvas de aprendizaje: dos graficos separados para mayor claridad
# Grafico 1: variantes SGD (arquitectura x alpha) - color unico por curva
sgd_res  = [r for r in resultados if r['grupo'] == 'SGD']
opt_res  = [r for r in resultados if r['grupo'] != 'SGD']

paleta_sgd = plt.cm.tab20(np.linspace(0, 1, len(sgd_res)))
fig, ax = plt.subplots(figsize=(13, 5))
for r, color in zip(sgd_res, paleta_sgd):
    ax.plot(r['hist_val'], label=r['nombre'], color=color, alpha=0.8, linewidth=1.2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Exactitud en test', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_title('Variantes SGD - arquitectura y tasa de aprendizaje', fontsize=12)
ax.legend(fontsize=7, loc='lower right', ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_curvas_sgd.png'), dpi=150)
plt.show()

# Grafico 2: comparacion de optimizadores (SGD vs Momentum vs Adam)
colores_opt = {'SGD': 'steelblue', 'MOMENTUM': 'crimson', 'ADAM': 'seagreen'}
fig, ax = plt.subplots(figsize=(11, 5))
for r in opt_res:
    ax.plot(r['hist_val'], label=r['nombre'],
            color=colores_opt.get(r['grupo'], 'gray'), linewidth=1.8, alpha=0.85)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Exactitud en test', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_title('Comparacion de optimizadores (SGD / Momentum / Adam)', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_curvas_optimizadores.png'), dpi=150)
plt.show()


# Reporte y matriz de confusion del mejor modelo
y_pred = mejor['modelo'].predict(X_test)
acc, precisiones, recalls, f1s, cm = reporte_metricas(y_test, y_pred)
imprimir_reporte(acc, precisiones, recalls, f1s, nombre=mejor['nombre'])

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel('Prediccion', fontsize=12); ax.set_ylabel('Real', fontsize=12)
ax.set_title(f'Matriz de confusion - {mejor["nombre"]}', fontsize=11)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_confusion.png'), dpi=150)
plt.show()

print("\nGraficos guardados: plot_curvas.png | plot_confusion.png")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlp import MLP

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


# ── Métricas ──────────────────────────────────────────────────────────────────

def matriz_confusion(y_true, y_pred, n_clases=10):
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def reporte_metricas(y_true, y_pred, n_clases=10):
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


# ── Helpers de IC ─────────────────────────────────────────────────────────────

def _rolling_std(arr, window=5):
    """Std local en ventana deslizante — indica la variabilidad época a época."""
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = np.std(arr[lo:hi])
    return out

def _bootstrap_ci_acc(y_true, y_pred, n=200, seed=42):
    """Bootstrap 95 % CI para la exactitud global (escalar)."""
    rng    = np.random.default_rng(seed)
    n_samp = len(y_true)
    accs   = []
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        accs.append(np.mean(y_pred[idx] == y_true[idx]))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

def _bootstrap_ci_metricas_digito(y_true, y_pred, n=200, seed=42):
    """Bootstrap 95 % CI para precisión/recall/F1 por dígito."""
    rng    = np.random.default_rng(seed)
    n_samp = len(y_true)
    prec_b, rec_b, f1_b = [], [], []
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        _, p, r, f, _ = reporte_metricas(y_true[idx], y_pred[idx])
        prec_b.append(p); rec_b.append(r); f1_b.append(f)
    prec_b = np.array(prec_b); rec_b = np.array(rec_b); f1_b = np.array(f1_b)
    return (np.percentile(prec_b, 2.5, 0), np.percentile(prec_b, 97.5, 0),
            np.percentile(rec_b,  2.5, 0), np.percentile(rec_b,  97.5, 0),
            np.percentile(f1_b,   2.5, 0), np.percentile(f1_b,   97.5, 0))


# ── Variantes arquitectura + alpha (SGD) ──────────────────────────────────────

VARIANTES = [
    {"capas": [128],      "alpha": 0.1,  "epochs": 50},
    {"capas": [128, 64],  "alpha": 0.1,  "epochs": 50},
    {"capas": [256, 128], "alpha": 0.1,  "epochs": 50},
    {"capas": [64,  32],  "alpha": 0.05, "epochs": 50},
    {"capas": [128],      "alpha": 0.05, "epochs": 50},
    {"capas": [128, 64],  "alpha": 0.05, "epochs": 50},
    {"capas": [256, 128], "alpha": 0.05, "epochs": 50},
]

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
        "nombre":   nombre,
        "modelo":   modelo,
        "hist_tr":  hist_tr,
        "hist_val": hist_val,
        "acc_test": modelo.score(X_test, y_test),
        "grupo":    "SGD",
    })

VARIANTES_OPT = [
    {"capas": [128, 64],  "alpha": 0.1,  "opt": "sgd",      "epochs": 50},
    {"capas": [256, 128], "alpha": 0.1,  "opt": "sgd",      "epochs": 50},
    {"capas": [128, 64],  "alpha": 0.1,  "opt": "momentum", "epochs": 50},
    {"capas": [256, 128], "alpha": 0.1,  "opt": "momentum", "epochs": 50},
    {"capas": [128, 64],  "alpha": 0.01, "opt": "adam",     "epochs": 50},
    {"capas": [256, 128], "alpha": 0.01, "opt": "adam",     "epochs": 50},
]

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
        "nombre":   nombre,
        "modelo":   modelo,
        "hist_tr":  hist_tr,
        "hist_val": hist_val,
        "acc_test": modelo.score(X_test, y_test),
        "grupo":    v['opt'].upper(),
    })


# ── Resumen comparativo ───────────────────────────────────────────────────────

print("Resumen comparativo de todas las variantes")
print(f"{'Variante':<50} {'Acc test %'}")
for r in resultados:
    print(f"{r['nombre']:<50} {r['acc_test']*100:.2f}%")

mejor = max(resultados, key=lambda r: r['acc_test'])
print(f"\nMejor modelo: {mejor['nombre']}  ->  {mejor['acc_test']*100:.2f}%")


# ── Gráfico 1 — Curvas SGD con banda de variabilidad ─────────────────────────

sgd_res  = [r for r in resultados if r['grupo'] == 'SGD']
opt_res  = [r for r in resultados if r['grupo'] != 'SGD']

paleta_sgd   = plt.cm.tab20(np.linspace(0, 1, len(sgd_res)))
n_epochs_sgd = len(sgd_res[0]['hist_val'])
ep_sgd       = np.arange(1, n_epochs_sgd + 1)

fig, ax = plt.subplots(figsize=(13, 5))
for r, color in zip(sgd_res, paleta_sgd):
    vals = np.array(r['hist_val'])
    std  = _rolling_std(vals)
    ax.fill_between(ep_sgd, vals - std, vals + std, color=color, alpha=0.10)
    ax.plot(ep_sgd, vals, label=r['nombre'], color=color, alpha=0.85, linewidth=1.2)
ax.set_xlim(10, n_epochs_sgd)
ax.set_xlabel('Época', fontsize=11)
ax.set_ylabel('Exactitud en prueba', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_title('Variantes SGD — arquitectura y tasa de aprendizaje (épocas 10–50)\n'
             '(banda = variabilidad local ±1σ ventana 5 épocas)', fontsize=12)
ax.legend(fontsize=7, loc='lower right', ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_curvas_sgd.png'), dpi=150)
#plt.show()


# ── Gráfico 2 — Comparación optimizadores con banda ──────────────────────────

colores_4    = ['crimson', 'tomato', 'seagreen', 'mediumaquamarine',
                'steelblue', 'cornflowerblue']
n_epochs_opt = len(opt_res[0]['hist_val'])
ep_opt       = np.arange(1, n_epochs_opt + 1)

fig, ax = plt.subplots(figsize=(11, 5))
for r, color in zip(opt_res, colores_4):
    vals = np.array(r['hist_val'])
    std  = _rolling_std(vals)
    ax.fill_between(ep_opt, vals - std, vals + std, color=color, alpha=0.12)
    ax.plot(ep_opt, vals, label=r['nombre'], color=color, linewidth=1.8, alpha=0.9)
ax.set_xlabel('Época', fontsize=11)
ax.set_ylabel('Exactitud en prueba', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_title('Comparación de optimizadores — Momentum vs Adam\n'
             '(2 arquitecturas × 2 optimizadores | banda = variabilidad local ±1σ)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_curvas_optimizadores.png'), dpi=150)
#plt.show()


# ── Gráfico 3 — Matrices de confusión top 4 ───────────────────────────────────

y_pred_mejor = mejor['modelo'].predict(X_test)
acc, precisiones, recalls, f1s, cm = reporte_metricas(y_test, y_pred_mejor)
imprimir_reporte(acc, precisiones, recalls, f1s, nombre=mejor['nombre'])

top4_conf = sorted(resultados, key=lambda r: r['acc_test'], reverse=True)[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Matrices de Confusión — Top 4 modelos (ordenados por exactitud)',
             fontsize=14, fontweight='bold')

for ax, r in zip(axes.flat, top4_conf):
    y_p = r['modelo'].predict(X_test)
    _, _, _, _, cm_r = reporte_metricas(y_test, y_p)
    im = ax.imshow(cm_r, cmap='Blues')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f"{r['nombre']}\nExactitud = {r['acc_test']*100:.1f}%",
                 fontsize=9, fontweight='bold')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm_r[i, j], ha='center', va='center', fontsize=7,
                    color='white' if cm_r[i, j] > cm_r.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_confusion.png'), dpi=150)
#plt.show()

print("\nGraficos guardados: plot_curvas_sgd.png | plot_curvas_optimizadores.png | plot_confusion.png")


# ── Gráfico 4 — Comparación exactitud final con IC bootstrap ─────────────────

from matplotlib.patches import Patch

todos_ordenados  = sorted(resultados, key=lambda r: r['acc_test'])
colores_grupo    = {'SGD': 'steelblue', 'MOMENTUM': 'crimson', 'ADAM': 'seagreen'}
colores_barras   = [colores_grupo.get(r['grupo'], 'gray') for r in todos_ordenados]
accs_finales     = [r['acc_test'] * 100 for r in todos_ordenados]

# Bootstrap CI para cada modelo
print("\nComputando IC bootstrap para las exactitudes finales...")
rng = np.random.default_rng(42)
n_samp = len(y_test)
xerr_lo, xerr_hi = [], []
for r in todos_ordenados:
    y_pred_r = r['modelo'].predict(X_test)
    accs_b = []
    for _ in range(200):
        idx = rng.integers(0, n_samp, n_samp)
        accs_b.append(np.mean(y_pred_r[idx] == y_test[idx]) * 100)
    lo = np.percentile(accs_b, 2.5)
    hi = np.percentile(accs_b, 97.5)
    acc_m = r['acc_test'] * 100
    xerr_lo.append(acc_m - lo)
    xerr_hi.append(hi - acc_m)

fig4, ax4 = plt.subplots(figsize=(11, 7))
barras = ax4.barh(range(len(todos_ordenados)), accs_finales,
                  color=colores_barras, alpha=0.85,
                  xerr=[xerr_lo, xerr_hi],
                  error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': 'dimgray'})
ax4.set_yticks(range(len(todos_ordenados)))
ax4.set_yticklabels([r['nombre'] for r in todos_ordenados], fontsize=8)
ax4.set_xlabel('Exactitud en prueba (%)', fontsize=11)
ax4.set_title('Comparación de exactitud final — todos los modelos\n'
              '(ordenados de menor a mayor | barras de error = IC 95 % bootstrap)',
              fontsize=12, fontweight='bold')
for i, (b, acc) in enumerate(zip(barras, accs_finales)):
    ax4.text(acc + xerr_hi[i] + 0.2, i, f'{acc:.1f}%', va='center', fontsize=8)
ax4.legend(handles=[Patch(color=c, label=g) for g, c in colores_grupo.items()], fontsize=10)
ax4.set_xlim(0, 108)
ax4.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_comparacion_modelos.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_comparacion_modelos.png")


# ── Gráfico 5 — Train vs Test top 4 con banda de variabilidad ────────────────

top4 = sorted(resultados, key=lambda r: r['acc_test'], reverse=True)[:4]

fig5, axes5 = plt.subplots(2, 2, figsize=(14, 8))
fig5.suptitle('Top 4 modelos — Entrenamiento vs Prueba (detección de sobreajuste)\n'
              '(banda = variabilidad local ±1σ ventana 5 épocas)',
              fontsize=13, fontweight='bold')

for ax, r in zip(axes5.flat, top4):
    ep = np.arange(1, len(r['hist_tr']) + 1)
    tr_v  = np.array(r['hist_tr'])  * 100
    val_v = np.array(r['hist_val']) * 100
    std_tr  = _rolling_std(tr_v)
    std_val = _rolling_std(val_v)

    ax.fill_between(ep, tr_v  - std_tr,  tr_v  + std_tr,  color='steelblue', alpha=0.15)
    ax.fill_between(ep, val_v - std_val, val_v + std_val, color='crimson',   alpha=0.15)
    ax.plot(ep, tr_v,  '-', color='steelblue', label='Entrenamiento', linewidth=1.5)
    ax.plot(ep, val_v, '-', color='crimson',   label='Prueba',        linewidth=1.5)

    gap = r['hist_tr'][-1]*100 - r['hist_val'][-1]*100
    ax.set_title(f"{r['nombre']}\n(brecha final: {gap:+.1f}%)", fontsize=9)
    ax.set_xlabel('Época', fontsize=9)
    ax.set_ylabel('Exactitud (%)', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_overfitting_top4.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_overfitting_top4.png")


# ── Gráfico 6 — Métricas por dígito con IC bootstrap ─────────────────────────

x_dig  = np.arange(10)
ancho  = 0.25

# Compute bootstrap CI
print("\nComputando IC bootstrap para métricas por dígito...")
prec_lo_b, prec_hi_b, rec_lo_b, rec_hi_b, f1_lo_b, f1_hi_b = \
    _bootstrap_ci_metricas_digito(y_test, y_pred_mejor)

prec_arr = np.array(precisiones)
rec_arr  = np.array(recalls)
f1_arr   = np.array(f1s)

prec_err = [prec_arr - prec_lo_b, prec_hi_b - prec_arr]
rec_err  = [rec_arr  - rec_lo_b,  rec_hi_b  - rec_arr]
f1_err   = [f1_arr   - f1_lo_b,   f1_hi_b   - f1_arr]

fig6, ax6 = plt.subplots(figsize=(12, 5))
ax6.bar(x_dig - ancho, precisiones, ancho, label='Precisión', color='steelblue', alpha=0.85,
        yerr=prec_err, error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax6.bar(x_dig,         recalls,     ancho, label='Recall',    color='crimson',   alpha=0.85,
        yerr=rec_err,  error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax6.bar(x_dig + ancho, f1s,         ancho, label='F1-Score',  color='seagreen',  alpha=0.85,
        yerr=f1_err,   error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax6.axhline(np.mean(f1s), color='black', linestyle='--', linewidth=1,
            label=f'Macro-F1 medio = {np.mean(f1s):.3f}')
ax6.set_title(f'Métricas por dígito — mejor modelo: {mejor["nombre"]}\n'
              f'(barras de error = IC 95 % bootstrap, 200 muestras)',
              fontsize=12, fontweight='bold')
ax6.set_xlabel('Dígito', fontsize=11)
ax6.set_ylabel('Valor de la métrica', fontsize=11)
ax6.set_xticks(x_dig)
ax6.set_xticklabels([str(i) for i in range(10)], fontsize=11)
ax6.set_ylim(0, 1.18)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_metricas_por_digito.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_metricas_por_digito.png")


# ── Gráfico 7 — Ejemplos mal clasificados ────────────────────────────────────

errores_idx = np.where(y_pred_mejor != y_test)[0]
n_mostrar   = min(20, len(errores_idx))
np.random.seed(0)
muestra     = np.random.choice(errores_idx, n_mostrar, replace=False)

fig7, axes7 = plt.subplots(4, 5, figsize=(11, 9))
fig7.suptitle(f'Ejemplos de errores — mejor modelo: {mejor["nombre"]}',
              fontsize=12, fontweight='bold')

for ax, idx in zip(axes7.flat, muestra):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray_r', interpolation='nearest')
    ax.set_title(f'Real: {y_test[idx]}  →  Pred: {y_pred_mejor[idx]}',
                 fontsize=8, color='crimson')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_errores.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_errores.png")

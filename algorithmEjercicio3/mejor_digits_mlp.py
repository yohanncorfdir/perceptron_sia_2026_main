import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mlp_v2 import MLPv2

DATA_DIR    = os.path.join(os.path.dirname(__file__), '../data')
MODELOS_DIR = os.path.join(os.path.dirname(__file__), 'modelos')
os.makedirs(MODELOS_DIR, exist_ok=True)


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar(path):
    df = pd.read_csv(path)
    X = np.array([np.array(ast.literal_eval(s), dtype=np.float32) for s in df['image']])
    y = df['label'].values.astype(int)
    return X, y

X_base,  y_base  = cargar(os.path.join(DATA_DIR, 'digits.csv'))
X_more,  y_more  = cargar(os.path.join(DATA_DIR, 'more_digits.csv'))
X_test,  y_test  = cargar(os.path.join(DATA_DIR, 'digits_test.csv'))

X_train = np.concatenate([X_base, X_more], axis=0)
y_train = np.concatenate([y_base, y_more], axis=0)

print(f"Train: {X_train.shape[0]} muestras")
print(f"Test : {X_test.shape[0]}  muestras")


# ── Métricas ──────────────────────────────────────────────────────────────────

def reporte(y_true, y_pred, nombre="", silent=False):
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
    if not silent:
        print(f"Metricas: {nombre}")
        print(f"Exactitud: {acc*100:.2f}%  |  Macro-F1: {np.mean(f1s):.3f}")
    return acc, cm

def metricas_por_clase(cm):
    precisiones, recalls, f1s = [], [], []
    for c in range(10):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisiones.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return precisiones, recalls, f1s

def graficar_confusion_simple(ax, cm, titulo):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel('Predicción', fontsize=9)
    ax.set_ylabel('Etiqueta real', fontsize=9)
    ax.set_title(titulo, fontsize=9, fontweight='bold')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=6,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    return im


# ── Helpers de IC ─────────────────────────────────────────────────────────────

def _rolling_std(arr, window=5):
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        out[i] = np.std(arr[lo:hi])
    return out

def _bootstrap_ci_acc(y_true, y_pred, n=200, seed=42):
    rng = np.random.default_rng(seed)
    n_samp = len(y_true)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        accs.append(np.mean(y_pred[idx] == y_true[idx]) * 100)
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

def _bootstrap_ci_f1_clase(y_true, y_pred, n=200, seed=42):
    rng = np.random.default_rng(seed)
    n_samp = len(y_true)
    f1_b = []
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        _, _, f1s_b = metricas_por_clase_raw(y_true[idx], y_pred[idx])
        f1_b.append(f1s_b)
    f1_b = np.array(f1_b)
    return np.percentile(f1_b, 2.5, 0), np.percentile(f1_b, 97.5, 0)

def metricas_por_clase_raw(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return metricas_por_clase(cm)


# ── Ruta de modelo ────────────────────────────────────────────────────────────

def _ruta_modelo(nombre):
    safe = nombre.strip().replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
    return os.path.join(MODELOS_DIR, safe + ".npz")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Variantes base (sin augmentación)
# ══════════════════════════════════════════════════════════════════════════════

VARIANTES = [
    {"nombre": "Ref: sigmoid [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "sigmoid",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    {"nombre": "ReLU      [256,128] adam  sin aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},

    {"nombre": "ReLU      [512,256] adam  sin aug",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_train, "y": y_train, "epochs": 80},
]

resultados = []
for v in VARIANTES:
    print(f"\n>>> {v['nombre']}")
    ruta = _ruta_modelo(v['nombre'])

    if os.path.exists(ruta):
        print(f"    [cargando desde {ruta}]")
        modelo = MLPv2.cargar(ruta)
        hist_tr = hist_val = hist_tr_loss = hist_val_loss = []
    else:
        np.random.seed(42)
        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])
        hist_tr, hist_val, hist_tr_loss, hist_val_loss = modelo.fit(
            v['X'], v['y'],
            epochs=v['epochs'], batch_size=64,
            X_val=X_test, y_val=y_test,
            verbose=True, paciencia=10, min_delta=0.001
        )
        modelo.guardar(ruta)
        print(f"    [modelo guardado en {ruta}]")

    acc_test = modelo.score(X_test, y_test)
    resultados.append({
        "nombre": v['nombre'], "modelo": modelo,
        "hist_tr": hist_tr, "hist_val": hist_val,
        "hist_tr_loss": hist_tr_loss, "hist_val_loss": hist_val_loss,
        "acc_test": acc_test,
    })


# ── Resumen sección 1 ─────────────────────────────────────────────────────────

print("Resumen ")
print(f"{'Variante':<45} {'Acc test %'}")
for r in resultados:
    marca = " <-- MEJOR" if r == max(resultados, key=lambda x: x['acc_test']) else ""
    print(f"{r['nombre']:<45} {r['acc_test']*100:.2f}%{marca}")

mejor = max(resultados, key=lambda r: r['acc_test'])
acc_final, cm = reporte(y_test, mejor['modelo'].predict(X_test), nombre=mejor['nombre'])
resultados_base = resultados


# ── GRÁFICOS SECCIÓN 1 ────────────────────────────────────────────────────────

DIR          = os.path.dirname(__file__)
colores_base = ['steelblue', 'crimson', 'seagreen']

# Gráfico 1 — Curvas de aprendizaje (exactitud)
fig1, ax1 = plt.subplots(figsize=(11, 5))
for r, color in zip(resultados_base, colores_base):
    if r['hist_val']:
        ep  = np.arange(1, len(r['hist_val']) + 1)
        v   = np.array(r['hist_val']) * 100
        std = _rolling_std(v)
        ax1.fill_between(ep, v - std, v + std, color=color, alpha=0.12)
        ax1.plot(ep, v, '-', color=color, linewidth=1.8,
                 label=f"{r['nombre']} — prueba ({r['acc_test']*100:.1f}%)")
        if r['hist_tr']:
            vt = np.array(r['hist_tr']) * 100
            ax1.plot(ep, vt, '--', color=color, linewidth=1, alpha=0.45,
                     label=f"{r['nombre']} — entreno")
    else:
        ax1.axhline(r['acc_test'] * 100, color=color, linestyle=':', linewidth=1.5,
                    label=f"{r['nombre']} ({r['acc_test']*100:.1f}%) — cargado de caché")
ax1.axhline(98, color='black', linestyle='--', linewidth=1, label='Objetivo 98 %')
ax1.set_xlabel('Época', fontsize=11)
ax1.set_ylabel('Exactitud (%)', fontsize=11)
ax1.set_title('Variantes base (sin augmentación) — Curvas de exactitud\n'
              '(banda = variabilidad local ±1σ | continua = prueba, discontinua = entreno)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_curvas_base.png'), dpi=150)
plt.show()

# Gráfico 1b — Curvas de pérdida (cross-entropy)
fig1b, ax1b = plt.subplots(figsize=(11, 5))
tiene_loss = False
for r, color in zip(resultados_base, colores_base):
    if r['hist_val_loss']:
        tiene_loss = True
        ep    = np.arange(1, len(r['hist_val_loss']) + 1)
        vl    = np.array(r['hist_val_loss'])
        vtl   = np.array(r['hist_tr_loss'])
        std_v = _rolling_std(vl)
        ax1b.fill_between(ep, vl - std_v, vl + std_v, color=color, alpha=0.12)
        ax1b.plot(ep, vl,  '-',  color=color, linewidth=1.8,
                  label=f"{r['nombre']} — prueba")
        ax1b.plot(ep, vtl, '--', color=color, linewidth=1, alpha=0.45,
                  label=f"{r['nombre']} — entreno")
    else:
        ax1b.axhline(0, color=color, linestyle=':', linewidth=1,
                     label=f"{r['nombre']} — cargado de caché")
ax1b.set_xlabel('Época', fontsize=11)
ax1b.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax1b.set_title('Variantes base — Curvas de pérdida (cross-entropy)\n'
               '(banda = variabilidad local ±1σ | continua = prueba, discontinua = entreno)',
               fontsize=12, fontweight='bold')
ax1b.legend(fontsize=8, loc='upper right')
ax1b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_loss_base.png'), dpi=150)
plt.show()

# Gráfico 2 — Comparación exactitud final con IC bootstrap
fig2, ax2 = plt.subplots(figsize=(9, 3.5))
accs_b = [r['acc_test'] * 100 for r in resultados_base]
xerr_lo2, xerr_hi2 = [], []
for r in resultados_base:
    y_pred_r = r['modelo'].predict(X_test)
    lo, hi   = _bootstrap_ci_acc(y_test, y_pred_r)
    xerr_lo2.append(r['acc_test'] * 100 - lo)
    xerr_hi2.append(hi - r['acc_test'] * 100)
barras2 = ax2.barh(range(len(resultados_base)), accs_b, color=colores_base, alpha=0.85,
                   xerr=[xerr_lo2, xerr_hi2],
                   error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': 'dimgray'})
ax2.set_yticks(range(len(resultados_base)))
ax2.set_yticklabels([r['nombre'] for r in resultados_base], fontsize=9)
ax2.set_xlabel('Exactitud en prueba (%)', fontsize=11)
ax2.set_title('Comparación exactitud final — variantes base\n(IC 95 % bootstrap)',
              fontsize=12, fontweight='bold')
ax2.set_xlim(min(accs_b) - 3, 103)
for b, acc, hi in zip(barras2, accs_b, xerr_hi2):
    ax2.text(acc + hi + 0.2, b.get_y() + b.get_height() / 2, f'{acc:.1f}%', va='center', fontsize=9)
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_comparacion_base.png'), dpi=150)
plt.show()

# Gráfico 3 — Matriz de confusión mejor modelo base
fig3, ax3 = plt.subplots(figsize=(8, 7))
im3 = graficar_confusion_simple(
    ax3, cm,
    f'Matriz de confusión — {mejor["nombre"]}\nExactitud = {acc_final*100:.1f}%'
)
plt.colorbar(im3, ax=ax3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_confusion_base.png'), dpi=150)
plt.show()

# Gráfico 4 — Métricas por dígito con IC bootstrap
prec_b, rec_b, f1s_b = metricas_por_clase(cm)
y_pred_mejor_base    = mejor['modelo'].predict(X_test)
n_samp_base          = len(y_test)

rng_m = np.random.default_rng(42)
prec_boot, rec_boot, f1_boot_d = [], [], []
for _ in range(200):
    idx = rng_m.integers(0, n_samp_base, n_samp_base)
    p_, r_, f_ = metricas_por_clase_raw(y_test[idx], y_pred_mejor_base[idx])
    prec_boot.append(p_); rec_boot.append(r_); f1_boot_d.append(f_)
prec_boot = np.array(prec_boot); rec_boot = np.array(rec_boot); f1_boot_d = np.array(f1_boot_d)

p_arr = np.array(prec_b); r_arr = np.array(rec_b); f_arr = np.array(f1s_b)
p_err = [p_arr - np.percentile(prec_boot, 2.5, 0), np.percentile(prec_boot, 97.5, 0) - p_arr]
r_err = [r_arr - np.percentile(rec_boot,  2.5, 0), np.percentile(rec_boot,  97.5, 0) - r_arr]
f_err = [f_arr - np.percentile(f1_boot_d, 2.5, 0), np.percentile(f1_boot_d, 97.5, 0) - f_arr]

x_dig = np.arange(10)
ancho = 0.25
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.bar(x_dig - ancho, prec_b, ancho, label='Precisión',  color='steelblue', alpha=0.85,
        yerr=p_err, error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax4.bar(x_dig,         rec_b,  ancho, label='Recall',     color='crimson',   alpha=0.85,
        yerr=r_err, error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax4.bar(x_dig + ancho, f1s_b,  ancho, label='F1-Score',   color='seagreen',  alpha=0.85,
        yerr=f_err, error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax4.axhline(np.mean(f1s_b), color='black', linestyle='--', linewidth=1,
            label=f'Macro-F1 = {np.mean(f1s_b):.3f}')
ax4.set_title(f'Métricas por dígito — {mejor["nombre"]}\n(IC 95 % bootstrap)',
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Dígito', fontsize=11)
ax4.set_ylabel('Valor de la métrica', fontsize=11)
ax4.set_xticks(x_dig); ax4.set_xticklabels([str(i) for i in range(10)], fontsize=11)
ax4.set_ylim(0, 1.15); ax4.legend(fontsize=10); ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_metricas_digito_base.png'), dpi=150)
plt.show()

print("\nGráficos guardados: plot_curvas_base | plot_loss_base | plot_comparacion_base | "
      "plot_confusion_base | plot_metricas_digito_base")


# ══════════════════════════════════════════════════════════════════════════════
# Augmentación de datos
# ══════════════════════════════════════════════════════════════════════════════

def augment(X, y, shifts=[(0,1),(0,-1),(1,0),(-1,0)]):
    X_aug, y_aug = [X], [y]
    for dr, dc in shifts:
        imgs = X.reshape(-1, 28, 28)
        imgs_s = np.roll(np.roll(imgs, shift=dr, axis=1), shift=dc, axis=2)
        X_aug.append(imgs_s.reshape(-1, 784))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)

X_aug, y_aug = augment(X_train, y_train)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Variantes CON augmentación
# ══════════════════════════════════════════════════════════════════════════════

VARIANTES_Aug = [
    {"nombre": "ReLU      [256,128] adam  CON aug",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "X": X_aug, "y": y_aug, "epochs": 80},

    {"nombre": "ReLU      [512,256] adam  CON aug + decay",
     "capas": [512, 256], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.005, "X": X_aug, "y": y_aug, "epochs": 80},
]

resultados = []
for v in VARIANTES_Aug:
    print(f"\n>>> {v['nombre']}")
    ruta = _ruta_modelo("aug_" + v['nombre'])

    if os.path.exists(ruta):
        print(f"    [cargando desde {ruta}]")
        modelo = MLPv2.cargar(ruta)
        hist_tr = hist_val = hist_tr_loss = hist_val_loss = []
    else:
        np.random.seed(42)
        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])
        hist_tr, hist_val, hist_tr_loss, hist_val_loss = modelo.fit(
            v['X'], v['y'],
            epochs=v['epochs'], batch_size=64,
            X_val=X_test, y_val=y_test,
            verbose=True, paciencia=10, min_delta=0.001
        )
        modelo.guardar(ruta)
        print(f"    [modelo guardado en {ruta}]")

    acc_test = modelo.score(X_test, y_test)
    resultados.append({
        "nombre": v['nombre'], "modelo": modelo,
        "hist_tr": hist_tr, "hist_val": hist_val,
        "hist_tr_loss": hist_tr_loss, "hist_val_loss": hist_val_loss,
        "acc_test": acc_test,
    })

resultados_aug = resultados
mejor_aug      = max(resultados_aug, key=lambda r: r['acc_test'])
_, cm_aug      = reporte(y_test, mejor_aug['modelo'].predict(X_test), nombre=mejor_aug['nombre'])


# ── GRÁFICOS SECCIÓN 2 ────────────────────────────────────────────────────────

colores_aug = ['darkorange', 'purple']

# Gráfico 5 — Curvas exactitud aug con banda
fig5, ax5 = plt.subplots(figsize=(11, 5))
for r, color in zip(resultados_aug, colores_aug):
    if r['hist_val']:
        ep  = np.arange(1, len(r['hist_val']) + 1)
        v   = np.array(r['hist_val']) * 100
        std = _rolling_std(v)
        ax5.fill_between(ep, v - std, v + std, color=color, alpha=0.12)
        ax5.plot(ep, v, '-', color=color, linewidth=1.8,
                 label=f"{r['nombre']} — prueba ({r['acc_test']*100:.1f}%)")
        if r['hist_tr']:
            vt = np.array(r['hist_tr']) * 100
            ax5.plot(ep, vt, '--', color=color, linewidth=1, alpha=0.45,
                     label=f"{r['nombre']} — entreno")
    else:
        ax5.axhline(r['acc_test'] * 100, color=color, linestyle=':', linewidth=1.5,
                    label=f"{r['nombre']} ({r['acc_test']*100:.1f}%) — cargado")
ax5.axhline(98, color='black', linestyle='--', linewidth=1, label='Objetivo 98 %')
ax5.set_xlabel('Época', fontsize=11); ax5.set_ylabel('Exactitud (%)', fontsize=11)
ax5.set_title('Variantes CON augmentación — Curvas de exactitud\n'
              '(banda = variabilidad local ±1σ)',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=8, loc='lower right'); ax5.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_curvas_aug.png'), dpi=150)
plt.show()

# Gráfico 5b — Curvas pérdida aug
fig5b, ax5b = plt.subplots(figsize=(11, 5))
for r, color in zip(resultados_aug, colores_aug):
    if r['hist_val_loss']:
        ep  = np.arange(1, len(r['hist_val_loss']) + 1)
        vl  = np.array(r['hist_val_loss'])
        vtl = np.array(r['hist_tr_loss'])
        std = _rolling_std(vl)
        ax5b.fill_between(ep, vl - std, vl + std, color=color, alpha=0.12)
        ax5b.plot(ep, vl,  '-',  color=color, linewidth=1.8, label=f"{r['nombre']} — prueba")
        ax5b.plot(ep, vtl, '--', color=color, linewidth=1, alpha=0.45, label=f"{r['nombre']} — entreno")
    else:
        ax5b.axhline(0, color=color, linestyle=':', linewidth=1,
                     label=f"{r['nombre']} — cargado de caché")
ax5b.set_xlabel('Época', fontsize=11); ax5b.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax5b.set_title('Variantes CON augmentación — Curvas de pérdida\n(banda = variabilidad local ±1σ)',
               fontsize=12, fontweight='bold')
ax5b.legend(fontsize=8, loc='upper right'); ax5b.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_loss_aug.png'), dpi=150)
plt.show()

# Gráfico 6 — Comparación global base vs aug con IC bootstrap
todos_res   = resultados_base + resultados_aug
colores_tod = colores_base + colores_aug
todos_ord   = sorted(zip(todos_res, colores_tod), key=lambda x: x[0]['acc_test'])

fig6, ax6 = plt.subplots(figsize=(11, 5))
xerr_lo6, xerr_hi6 = [], []
for r, _ in todos_ord:
    y_pred_r = r['modelo'].predict(X_test)
    lo, hi   = _bootstrap_ci_acc(y_test, y_pred_r)
    xerr_lo6.append(r['acc_test'] * 100 - lo)
    xerr_hi6.append(hi - r['acc_test'] * 100)

for i, (r, color) in enumerate(todos_ord):
    acc_m = r['acc_test'] * 100
    ax6.barh(i, acc_m, color=color, alpha=0.85,
             xerr=[[xerr_lo6[i]], [xerr_hi6[i]]],
             error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': 'dimgray'})
    ax6.text(acc_m + xerr_hi6[i] + 0.1, i, f"{acc_m:.1f}%", va='center', fontsize=8)
ax6.set_yticks(range(len(todos_ord)))
ax6.set_yticklabels([r['nombre'] for r, _ in todos_ord], fontsize=8)
ax6.set_xlabel('Exactitud en prueba (%)', fontsize=11)
ax6.set_title('Comparación global — variantes base vs augmentación\n'
              '(ordenadas por exactitud | IC 95 % bootstrap)',
              fontsize=12, fontweight='bold')
ax6.legend(handles=[Patch(color='gray',      label='Variantes base'),
                    Patch(color='darkorange', label='CON augmentación')], fontsize=10)
ax6.set_xlim(min(r['acc_test'] * 100 for r, _ in todos_ord) - 3, 103)
ax6.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_comparacion_base_vs_aug.png'), dpi=150)
plt.show()

# Gráfico 7 — Matriz de confusión mejor aug
fig7, ax7 = plt.subplots(figsize=(8, 7))
im7 = graficar_confusion_simple(
    ax7, cm_aug,
    f'Matriz de confusión — {mejor_aug["nombre"]}\nExactitud = {mejor_aug["acc_test"]*100:.1f}%'
)
plt.colorbar(im7, ax=ax7)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_confusion_aug.png'), dpi=150)
plt.show()

print("\nGráficos guardados: plot_curvas_aug | plot_loss_aug | "
      "plot_comparacion_base_vs_aug | plot_confusion_aug")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfico especial — Todas las matrices de confusión en una figura
# ══════════════════════════════════════════════════════════════════════════════

fig_cm_all, axes_cm = plt.subplots(2, 3, figsize=(20, 13))
fig_cm_all.suptitle('Matrices de confusión — todas las variantes (base + augmentación)',
                    fontsize=14, fontweight='bold')

todos_para_cm = todos_res  # 3 base + 2 aug
for ax_cm, r in zip(axes_cm.flat[:5], todos_para_cm):
    y_p = r['modelo'].predict(X_test)
    _, cm_r = reporte(y_test, y_p, silent=True)
    im_cm = graficar_confusion_simple(
        ax_cm, cm_r,
        f"{r['nombre']}\nExact. = {r['acc_test']*100:.1f}%"
    )
    plt.colorbar(im_cm, ax=ax_cm)
axes_cm.flat[5].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_todas_confusiones.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_todas_confusiones.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — K-Fold Cross Validation
# ══════════════════════════════════════════════════════════════════════════════

K = 5

print(f"Stratified K-Fold Cross Validation | K={K} | Muestras aug: {len(X_aug)}")

# Índices estratificados por clase
folds_por_clase = []
np.random.seed(42)
for c in range(10):
    idx_c = np.where(y_aug == c)[0]
    np.random.shuffle(idx_c)
    folds_por_clase.append(np.array_split(idx_c, K))

folds = [np.concatenate([folds_por_clase[c][k] for c in range(10)]) for k in range(K)]

VARIANTES_CrossVal = [
    {"nombre": "ReLU      [256,128] adam  CON aug + CV",
     "capas": [256, 128], "alpha": 0.001, "opt": "adam", "act": "relu",
     "decay": 0.0, "epochs": 80},

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
        idx_test_cv  = folds[k]
        idx_train_cv = np.concatenate([folds[i] for i in range(K) if i != k])
        X_train_cv, y_train_cv = X_aug[idx_train_cv], y_aug[idx_train_cv]
        X_test_cv,  y_test_cv  = X_aug[idx_test_cv],  y_aug[idx_test_cv]

        modelo = MLPv2(n_entrada=784, capas_ocultas=v['capas'], n_salida=10,
                       alpha=v['alpha'], optimizador=v['opt'],
                       activacion=v['act'], lr_decay=v['decay'])

        hist_tr, hist_val, hist_tr_loss, hist_val_loss = modelo.fit(
            X_train_cv, y_train_cv,
            epochs=v['epochs'], batch_size=64,
            X_val=X_test_cv, y_val=y_test_cv,
            verbose=False, paciencia=10, min_delta=0.001
        )

        acc_fold  = modelo.score(X_test_cv, y_test_cv)
        y_pred_cv = modelo.predict(X_test_cv)

        f1s_cv = []
        for c in range(10):
            tp = np.sum((y_pred_cv == c) & (y_test_cv == c))
            fp = np.sum((y_pred_cv == c) & (y_test_cv != c))
            fn = np.sum((y_pred_cv != c) & (y_test_cv == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s_cv.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        macro_f1 = np.mean(f1s_cv)

        accs_fold.append(acc_fold)
        print(f"{v['nombre']:<45} {k+1:<6} {acc_fold*100:<10.2f} {macro_f1:<10.3f}")
        resultados_cv.append({
            "nombre": v['nombre'], "fold": k,
            "acc_test": acc_fold, "macro_f1": macro_f1,
            "f1s_clase": f1s_cv,
            "hist_tr": hist_tr, "hist_val": hist_val,
            "hist_tr_loss": hist_tr_loss, "hist_val_loss": hist_val_loss,
        })

    media_cv = np.mean(accs_fold) * 100
    std_cv   = np.std(accs_fold) * 100
    print(f"  --> Media: {media_cv:.2f}%  |  Std: {std_cv:.2f}%\n")


# ── GRÁFICOS SECCIÓN 3 ────────────────────────────────────────────────────────

nombres_cv = list(dict.fromkeys(r['nombre'] for r in resultados_cv))
colores_cv = ['darkorange', 'purple']

# Gráfico 8 — Exactitud por fold con banda media ± std
fig8, ax8 = plt.subplots(figsize=(11, 5))
ancho_cv  = 0.35
x_folds   = np.arange(K)

for i, (nombre, color) in enumerate(zip(nombres_cv, colores_cv)):
    accs_v = [r['acc_test'] * 100 for r in resultados_cv if r['nombre'] == nombre]
    media  = np.mean(accs_v)
    std    = np.std(accs_v)
    offset = (i - 0.5) * ancho_cv
    ax8.bar(x_folds + offset, accs_v, ancho_cv, color=color, alpha=0.70,
            label=f'{nombre[:35]}… (μ={media:.1f}% ±{std:.1f}%)')
    # Banda media ± std
    ax8.axhspan(media - std, media + std, alpha=0.08, color=color)
    ax8.axhline(media, color=color, linestyle='--', linewidth=1.5, alpha=0.85)

ax8.set_xlabel('Fold', fontsize=11)
ax8.set_ylabel('Exactitud (%)', fontsize=11)
ax8.set_xticks(x_folds)
ax8.set_xticklabels([f'Fold {k+1}' for k in range(K)], fontsize=10)
ax8.set_title('K-Fold Cross Validation — Exactitud por fold\n'
              '(discontinua = media | banda coloreada = IC ±1σ entre folds)',
              fontsize=12, fontweight='bold')
ax8.legend(fontsize=8, loc='lower right')
ax8.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_kfold_exactitud.png'), dpi=150)
plt.show()

# Gráfico 9 — Curvas de aprendizaje K-Fold: media ± std entre folds
fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5))
fig9.suptitle('K-Fold — Curvas de aprendizaje: media ± std entre folds',
              fontsize=13, fontweight='bold')

paleta_folds = plt.cm.tab10(np.linspace(0, 0.5, K))

for ax_cv, nombre, color_cv in zip(axes9, nombres_cv, colores_cv):
    fold_curves = [np.array(r['hist_val']) * 100 for r in resultados_cv
                   if r['nombre'] == nombre and r['hist_val']]
    if fold_curves:
        max_len = max(len(c) for c in fold_curves)
        padded  = np.array([np.pad(c, (0, max_len - len(c)), 'edge') for c in fold_curves])
        mean_ep = padded.mean(0)
        std_ep  = padded.std(0)
        ep      = np.arange(1, max_len + 1)
        ax_cv.fill_between(ep, mean_ep - std_ep, mean_ep + std_ep,
                           color=color_cv, alpha=0.20, label='IC ±1σ entre folds')
        ax_cv.plot(ep, mean_ep, color=color_cv, linewidth=2.5, label='Media folds')
        for r in resultados_cv:
            if r['nombre'] == nombre and r['hist_val']:
                ep_f = np.arange(1, len(r['hist_val']) + 1)
                ax_cv.plot(ep_f, [v * 100 for v in r['hist_val']],
                           color=paleta_folds[r['fold']], linewidth=1, alpha=0.45,
                           label=f"Fold {r['fold']+1}")
    ax_cv.axhline(98, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Objetivo 98 %')
    ax_cv.set_title(nombre[:50], fontsize=9, fontweight='bold')
    ax_cv.set_xlabel('Época', fontsize=10)
    ax_cv.set_ylabel('Exactitud en prueba (%)', fontsize=10)
    ax_cv.legend(fontsize=7, loc='lower right')
    ax_cv.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_kfold_curvas.png'), dpi=150)
plt.show()

# Gráfico 9b — Curvas de pérdida K-Fold
fig9b, axes9b = plt.subplots(1, 2, figsize=(14, 5))
fig9b.suptitle('K-Fold — Curvas de pérdida (cross-entropy): media ± std entre folds',
               fontsize=13, fontweight='bold')

for ax_cv, nombre, color_cv in zip(axes9b, nombres_cv, colores_cv):
    loss_curves = [np.array(r['hist_val_loss']) for r in resultados_cv
                   if r['nombre'] == nombre and r['hist_val_loss']]
    if loss_curves:
        max_len  = max(len(c) for c in loss_curves)
        padded   = np.array([np.pad(c, (0, max_len - len(c)), 'edge') for c in loss_curves])
        mean_l   = padded.mean(0)
        std_l    = padded.std(0)
        ep       = np.arange(1, max_len + 1)
        ax_cv.fill_between(ep, mean_l - std_l, mean_l + std_l,
                           color=color_cv, alpha=0.20, label='IC ±1σ entre folds')
        ax_cv.plot(ep, mean_l, color=color_cv, linewidth=2.5, label='Media folds')
        for r in resultados_cv:
            if r['nombre'] == nombre and r['hist_val_loss']:
                ep_f = np.arange(1, len(r['hist_val_loss']) + 1)
                ax_cv.plot(ep_f, r['hist_val_loss'],
                           color=paleta_folds[r['fold']], linewidth=1, alpha=0.45,
                           label=f"Fold {r['fold']+1}")
    ax_cv.set_title(nombre[:50], fontsize=9, fontweight='bold')
    ax_cv.set_xlabel('Época', fontsize=10)
    ax_cv.set_ylabel('Cross-Entropy Loss', fontsize=10)
    ax_cv.legend(fontsize=7, loc='upper right')
    ax_cv.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_kfold_loss.png'), dpi=150)
plt.show()

print("\nGráficos guardados: plot_kfold_exactitud | plot_kfold_curvas | plot_kfold_loss")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfico comparativo — F1 por dígito: base vs aug vs CV (mejor fold)
# ══════════════════════════════════════════════════════════════════════════════

# Mejor modelo base
y_pred_base_best = mejor['modelo'].predict(X_test)
_, _, f1s_base_best = metricas_por_clase_raw(y_test, y_pred_base_best)

# Mejor modelo aug
y_pred_aug_best = mejor_aug['modelo'].predict(X_test)
_, _, f1s_aug_best = metricas_por_clase_raw(y_test, y_pred_aug_best)

# F1 medio ± std por dígito (K-Fold, variante con mayor exactitud media)
cv_mean_acc = {n: np.mean([r['acc_test'] for r in resultados_cv if r['nombre'] == n])
               for n in nombres_cv}
mejor_var_cv = max(cv_mean_acc, key=cv_mean_acc.get)
f1s_cv_por_fold = np.array([r['f1s_clase'] for r in resultados_cv if r['nombre'] == mejor_var_cv])
f1s_cv_mean = f1s_cv_por_fold.mean(0)
f1s_cv_std  = f1s_cv_por_fold.std(0)

x_dig = np.arange(10)
ancho = 0.22

# Bootstrap CI para base y aug
f1_lo_base, f1_hi_base = _bootstrap_ci_f1_clase(y_test, y_pred_base_best)
f1_lo_aug,  f1_hi_aug  = _bootstrap_ci_f1_clase(y_test, y_pred_aug_best)
f1_arr_base = np.array(f1s_base_best)
f1_arr_aug  = np.array(f1s_aug_best)

fig_f1_comp, ax_f1 = plt.subplots(figsize=(14, 6))
ax_f1.bar(x_dig - ancho,    f1s_base_best, ancho, label=f'Base (mejor): {mejor["nombre"][:25]}…',
          color='steelblue', alpha=0.85,
          yerr=[f1_arr_base - f1_lo_base, f1_hi_base - f1_arr_base],
          error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax_f1.bar(x_dig,             f1s_aug_best,  ancho, label=f'Aug (mejor): {mejor_aug["nombre"][:25]}…',
          color='darkorange', alpha=0.85,
          yerr=[f1_arr_aug - f1_lo_aug, f1_hi_aug - f1_arr_aug],
          error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax_f1.bar(x_dig + ancho,     f1s_cv_mean,   ancho, label=f'CV media (±1σ folds): {mejor_var_cv[:25]}…',
          color='seagreen',   alpha=0.85,
          yerr=f1s_cv_std,
          error_kw={'elinewidth': 1.0, 'capsize': 3, 'ecolor': 'dimgray'})
ax_f1.set_title('Comparación F1 por dígito — Base vs Augmentación vs K-Fold\n'
                '(IC 95 % bootstrap para base/aug | IC ±1σ entre folds para CV)',
                fontsize=12, fontweight='bold')
ax_f1.set_xlabel('Dígito', fontsize=11)
ax_f1.set_ylabel('F1-Score', fontsize=11)
ax_f1.set_xticks(x_dig)
ax_f1.set_xticklabels([str(i) for i in range(10)], fontsize=11)
ax_f1.set_ylim(0, 1.15)
ax_f1.legend(fontsize=9)
ax_f1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_f1_comparacion_modelos.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_f1_comparacion_modelos.png")


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard final — resumen visual de la progresión de exactitud
# ══════════════════════════════════════════════════════════════════════════════

etapas   = ['Base\n(mejor)', 'Aug\n(mejor)', 'CV mean\n(mejor var.)']
accs_etapa = [mejor['acc_test'] * 100,
              mejor_aug['acc_test'] * 100,
              np.mean([r['acc_test'] for r in resultados_cv if r['nombre'] == mejor_var_cv]) * 100]
std_etapa = [0,
             0,
             np.std([r['acc_test'] for r in resultados_cv if r['nombre'] == mejor_var_cv]) * 100]

# Bootstrap CI para base y aug
lo_base, hi_base = _bootstrap_ci_acc(y_test, y_pred_base_best)
lo_aug,  hi_aug  = _bootstrap_ci_acc(y_test, y_pred_aug_best)
yerr_dash_lo = [accs_etapa[0] - lo_base, accs_etapa[1] - lo_aug, std_etapa[2]]
yerr_dash_hi = [hi_base - accs_etapa[0], hi_aug - accs_etapa[1], std_etapa[2]]

colores_etapa = ['steelblue', 'darkorange', 'seagreen']

fig_dash, ax_dash = plt.subplots(figsize=(9, 5))
barras_d = ax_dash.bar(etapas, accs_etapa, color=colores_etapa, alpha=0.85, width=0.45,
                       yerr=[yerr_dash_lo, yerr_dash_hi],
                       error_kw={'elinewidth': 2, 'capsize': 6, 'ecolor': 'dimgray'})
ax_dash.axhline(98, color='black', linestyle='--', linewidth=1.5, label='Objetivo 98 %')
for b, acc, lo, hi in zip(barras_d, accs_etapa, yerr_dash_lo, yerr_dash_hi):
    ax_dash.text(b.get_x() + b.get_width() / 2, acc + hi + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_dash.set_ylabel('Exactitud en prueba (%)', fontsize=11)
ax_dash.set_title('Progresión de exactitud a través de las etapas\n'
                  '(IC 95 % bootstrap para base/aug | IC ±1σ folds para CV)',
                  fontsize=12, fontweight='bold')
ax_dash.set_ylim(min(accs_etapa) - 5, 102)
ax_dash.legend(fontsize=10)
ax_dash.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_dashboard_progresion.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_dashboard_progresion.png")

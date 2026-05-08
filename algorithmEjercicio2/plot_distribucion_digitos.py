import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
DIR      = os.path.dirname(__file__)


# ── Carga ─────────────────────────────────────────────────────────────────────

df_train = pd.read_csv(os.path.join(DATA_DIR, 'digits.csv'))
df_test  = pd.read_csv(os.path.join(DATA_DIR, 'digits_test.csv'))

conteo_train = df_train['label'].value_counts().reindex(range(10), fill_value=0)
conteo_test  = df_test['label'].value_counts().reindex(range(10), fill_value=0)

digitos = np.arange(10)
total_train = len(df_train)
total_test  = len(df_test)

print(f"Total entrenamiento : {total_train} imágenes")
print(f"Total prueba        : {total_test}  imágenes")
print()
print(f"{'Dígito':<10} {'Train':>8} {'Train %':>9} {'Test':>8} {'Test %':>9}")
print("-" * 50)
for d in digitos:
    ct = conteo_train.get(d, 0)
    cv = conteo_test.get(d, 0)
    print(f"{d:<10} {ct:>8} {ct/total_train*100:>8.1f}%  {cv:>8} {cv/total_test*100:>8.1f}%")


# ── Gráfico ───────────────────────────────────────────────────────────────────

ancho = 0.38
x = digitos

fig, ax = plt.subplots(figsize=(11, 5))

barras_tr = ax.bar(x - ancho / 2, conteo_train.values, ancho,
                   label=f'Entrenamiento (n={total_train})',
                   color='steelblue', alpha=0.85)
barras_te = ax.bar(x + ancho / 2, conteo_test.values, ancho,
                   label=f'Prueba (n={total_test})',
                   color='darkorange', alpha=0.85)

# Etiquetas encima de cada barra
for b in barras_tr:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
            str(int(b.get_height())), ha='center', va='bottom', fontsize=8)
for b in barras_te:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
            str(int(b.get_height())), ha='center', va='bottom', fontsize=8)


ax.set_title('Distribución de dígitos — digits.csv\n'
             '(entrenamiento vs prueba)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Dígito', fontsize=11)
ax.set_ylabel('Número de imágenes', fontsize=11)
ax.set_xticks(digitos)
ax.set_xticklabels([str(d) for d in digitos], fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_distribucion_digitos.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_distribucion_digitos.png")

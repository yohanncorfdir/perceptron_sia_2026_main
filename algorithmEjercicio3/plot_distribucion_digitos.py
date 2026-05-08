import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
DIR      = os.path.dirname(__file__)


# ── Carga ─────────────────────────────────────────────────────────────────────

df_digits      = pd.read_csv(os.path.join(DATA_DIR, 'digits.csv'))
df_more_digits = pd.read_csv(os.path.join(DATA_DIR, 'more_digits.csv'))
df_test        = pd.read_csv(os.path.join(DATA_DIR, 'digits_test.csv'))

conteo_digits = df_digits['label'].value_counts().reindex(range(10), fill_value=0)
conteo_more   = df_more_digits['label'].value_counts().reindex(range(10), fill_value=0)
conteo_test   = df_test['label'].value_counts().reindex(range(10), fill_value=0)

n_digits = len(df_digits)
n_more   = len(df_more_digits)
n_train  = n_digits + n_more
n_test   = len(df_test)

print(f"digits.csv      : {n_digits} imágenes")
print(f"more_digits.csv : {n_more} imágenes")
print(f"Total train     : {n_train} imágenes")
print(f"digits_test.csv : {n_test}  imágenes")
print()
print(f"{'Dígito':<8} {'digits':>8} {'more':>8} {'train total':>12} {'test':>8}")
print("-" * 50)
for d in range(10):
    cd = conteo_digits[d]
    cm = conteo_more[d]
    ct = conteo_test[d]
    print(f"{d:<8} {cd:>8} {cm:>8} {cd+cm:>12} {ct:>8}")


# ── Gráfico ───────────────────────────────────────────────────────────────────

digitos = np.arange(10)
ancho   = 0.26
x       = digitos

fig, ax = plt.subplots(figsize=(13, 5))

barras_d = ax.bar(x - ancho, conteo_digits.values, ancho,
                  label=f'digits.csv (n={n_digits})',
                  color='steelblue', alpha=0.85)
barras_m = ax.bar(x,          conteo_more.values,   ancho,
                  label=f'more_digits.csv (n={n_more})',
                  color='seagreen', alpha=0.85)
barras_t = ax.bar(x + ancho,  conteo_test.values,   ancho,
                  label=f'digits_test.csv (n={n_test})',
                  color='darkorange', alpha=0.85)

# Etiquetas encima de cada barra
for b in barras_d:
    if b.get_height() > 0:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                str(int(b.get_height())), ha='center', va='bottom', fontsize=7)
for b in barras_m:
    if b.get_height() > 0:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                str(int(b.get_height())), ha='center', va='bottom', fontsize=7)
for b in barras_t:
    if b.get_height() > 0:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                str(int(b.get_height())), ha='center', va='bottom', fontsize=7)

# Suma total train (digits + more_digits) centrada entre las dos barras
conteo_total_train = conteo_digits.values + conteo_more.values
for d in digitos:
    total = conteo_total_train[d]
    x_centro = d - ancho / 2          # centro entre barra azul y verde
    y_pos    = max(conteo_digits[d], conteo_more[d]) + 60
    ax.text(x_centro, y_pos, f'sum {total}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')


ax.set_title('Distribución de dígitos\n'
             '(digits.csv  |  more_digits.csv  |  digits_test.csv)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Dígito', fontsize=11)
ax.set_ylabel('Número de imágenes', fontsize=11)
ax.set_xticks(digitos)
ax.set_xticklabels([str(d) for d in digitos], fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'plot_distribucion_digitos.png'), dpi=150)
plt.show()
print("Gráfico guardado: plot_distribucion_digitos.png")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

'''
Ese script permite una exploracion de datos 
Vamos a verificar potenciales corelaciones entre los datos 
Verificar si existen o no outliers
'''
# --- Carga de datos ---
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/fraud_dataset.csv'))

features = df.drop(columns=['flagged_fraud']).columns.tolist()
target   = 'flagged_fraud'

print(f"Conjunto de datos: {len(df)} muestras, {len(features)} variables")
print(f"Fraudes: {df[target].sum()} ({df[target].mean()*100:.1f}%)")
print(f"No fraudes: {(1-df[target]).sum()} ({(1-df[target]).mean()*100:.1f}%)\n")

# =============================================================
# MATRIZ DE CORRELACION
# =============================================================
corr = df[features + [target]].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))  # Solo triangulo inferior
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Matriz de correlacion - fraud_dataset", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_correlacion.png'), dpi=150)
plt.show()

# --- Identificamos las 2 features mas correlacionadas con la variable objetivo ---
corr_target = corr[target].drop(target).abs().sort_values(ascending=False)
feat1, feat2 = corr_target.index[0], corr_target.index[1]
print(f"Features mas correlacionadas con '{target}':")
print(f"  1. {feat1}  (r = {corr[target][feat1]:.3f})")
print(f"  2. {feat2}  (r = {corr[target][feat2]:.3f})\n")


# =============================================================
# BOXPLOTS: deteccion de outliers por variable
# =============================================================
n_cols = 3
n_rows = int(np.ceil(len(features) / n_cols))

fig = plt.figure(figsize=(n_cols * 5, n_rows * 3.5))
gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.35)

for idx, col in enumerate(features):
    ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
    data_plot = [no_fraud[col].values, fraud[col].values]
    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('crimson')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['No fraude', 'Fraude'], fontsize=8)
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis='y', labelsize=7)

    # Contamos outliers (puntos fuera de 1.5*IQR)
    outliers = sum(len(fl.get_ydata()) for fl in bp['fliers'])
    if outliers > 0:
        ax.set_xlabel(f"{outliers} outlier(s)", fontsize=7, color='darkorange')

fig.suptitle("Boxplots por variable: No fraude vs Fraude", fontsize=14, y=1.01)
plt.savefig(os.path.join(os.path.dirname(__file__), 'plot_boxplots.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print("Graficos guardados: plot_correlacion.png | plot_scatter.png | plot_boxplots.png")

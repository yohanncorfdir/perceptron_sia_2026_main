import numpy as np
import matplotlib.pyplot as plt


# ── Métricas ──────────────────────────────────────────────────────────────────

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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1

def bce(y_true, proba):
    """Entropía cruzada binaria (PerceptronNoLineal)."""
    p = np.clip(proba, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def tasa_error(y_true, y_pred):
    """Tasa de error / MSE binaria (PerceptronLineal)."""
    return np.mean((y_pred - y_true) ** 2)


# ── Helpers de intervalos de confianza ────────────────────────────────────────

def _bootstrap_ci_umbral(probas, y_true, thresholds, n=200, seed=42):
    """Bootstrap 95 % CI para las curvas precisión/recall/F1 vs umbral."""
    rng    = np.random.default_rng(seed)
    n_samp = len(y_true)
    prec_b, rec_b, f1_b = [], [], []
    for _ in range(n):
        idx  = rng.integers(0, n_samp, n_samp)
        p_r, r_r, f_r = [], [], []
        for t in thresholds:
            y_t = (probas[idx] >= t).astype(int)
            _, p, r, f = metricas(y_true[idx], y_t)
            p_r.append(p); r_r.append(r); f_r.append(f)
        prec_b.append(p_r); rec_b.append(r_r); f1_b.append(f_r)
    prec_b = np.array(prec_b); rec_b = np.array(rec_b); f1_b = np.array(f1_b)
    return (np.percentile(prec_b, 2.5, 0), np.percentile(prec_b, 97.5, 0),
            np.percentile(rec_b,  2.5, 0), np.percentile(rec_b,  97.5, 0),
            np.percentile(f1_b,   2.5, 0), np.percentile(f1_b,   97.5, 0))


def _bootstrap_ci_roc(probas, y_true, n=200, seed=42):
    """Bootstrap 95 % CI para la curva ROC sobre una rejilla FPR fija."""
    rng      = np.random.default_rng(seed)
    n_samp   = len(y_true)
    umbrales = np.linspace(0, 1, 300)
    fpr_grid = np.linspace(0, 1, 100)
    tpr_all  = []
    for _ in range(n):
        idx  = rng.integers(0, n_samp, n_samp)
        pb, yb = probas[idx], y_true[idx]
        tpr_l, fpr_l = [], []
        for t in umbrales:
            y_t  = (pb >= t).astype(int)
            tp_t = np.sum((y_t == 1) & (yb == 1))
            fn_t = np.sum((y_t == 0) & (yb == 1))
            fp_t = np.sum((y_t == 1) & (yb == 0))
            tn_t = np.sum((y_t == 0) & (yb == 0))
            tpr_l.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
            fpr_l.append(fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0)
        ord_b  = np.argsort(fpr_l)
        tpr_all.append(np.interp(fpr_grid, np.array(fpr_l)[ord_b], np.array(tpr_l)[ord_b]))
    tpr_mat = np.array(tpr_all)
    return fpr_grid, np.percentile(tpr_mat, 2.5, 0), np.percentile(tpr_mat, 97.5, 0)


# ── Helper interno ────────────────────────────────────────────────────────────

def _panel_confusion(ax, tp, tn, fp, fn, titulo):
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Etiqueta real', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Fraude (0)', 'Fraude (1)'], fontsize=9)
    ax.set_yticklabels(['No Fraude (0)', 'Fraude (1)'], fontsize=9)
    etiquetas = [['VN', 'FP'], ['FN', 'VP']]
    umbral_color = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > umbral_color else 'black'
            ax.text(j, i, f"{etiquetas[i][j]}\n{cm[i, j]}",
                    ha='center', va='center', fontsize=13, fontweight='bold', color=color)


# ── Gráficos públicos ─────────────────────────────────────────────────────────

def graficar_costo(epochs_range,
                   costo_lin_train, costo_lin_test,
                   costo_nolin_train, costo_nolin_test,
                   save_path,
                   std_lin_train=None, std_lin_test=None,
                   std_nolin_train=None, std_nolin_test=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Evolución de la función de costo — Detección de Fraude',
                 fontsize=14, fontweight='bold')

    clt  = np.array(costo_lin_train)
    clte = np.array(costo_lin_test)
    if std_lin_train is not None:
        s = np.array(std_lin_train)
        ax1.fill_between(epochs_range, clt - s, clt + s, color='steelblue', alpha=0.18,
                         label='IC ±1σ entreno')
    if std_lin_test is not None:
        s = np.array(std_lin_test)
        ax1.fill_between(epochs_range, clte - s, clte + s, color='crimson', alpha=0.18,
                         label='IC ±1σ prueba')
    ax1.plot(epochs_range, clt,  'o-', color='steelblue', label='Entrenamiento')
    ax1.plot(epochs_range, clte, 's-', color='crimson',   label='Prueba')
    ax1.set_title('Perceptrón Lineal\n(Función escalón — Tasa de Error)', fontsize=11)
    ax1.set_xlabel('Época', fontsize=10)
    ax1.set_ylabel('Tasa de Error', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    cnlt  = np.array(costo_nolin_train)
    cnlte = np.array(costo_nolin_test)
    if std_nolin_train is not None:
        s = np.array(std_nolin_train)
        ax2.fill_between(epochs_range, cnlt - s, cnlt + s, color='steelblue', alpha=0.18,
                         label='IC ±1σ entreno')
    if std_nolin_test is not None:
        s = np.array(std_nolin_test)
        ax2.fill_between(epochs_range, cnlte - s, cnlte + s, color='crimson', alpha=0.18,
                         label='IC ±1σ prueba')
    ax2.plot(epochs_range, cnlt,  'o-', color='steelblue', label='Entrenamiento')
    ax2.plot(epochs_range, cnlte, 's-', color='crimson',   label='Prueba')
    ax2.set_title('Perceptrón No Lineal\n(Función sigmoide — Entropía Cruzada Binaria)', fontsize=11)
    ax2.set_xlabel('Época', fontsize=10)
    ax2.set_ylabel('Entropía Cruzada Binaria', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_confusion(tp_lin, tn_lin, fp_lin, fn_lin, acc_lin, f1_lin, rec_lin,
                       tp_nol, tn_nol, fp_nol, fn_nol, acc_nol, f1_nol, rec_nol,
                       save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Matrices de Confusión — Conjunto de Prueba (20 %)',
                 fontsize=14, fontweight='bold')

    _panel_confusion(ax1, tp_lin, tn_lin, fp_lin, fn_lin,
                     f'Perceptrón Lineal\nAcc={acc_lin*100:.1f}%  F1={f1_lin:.3f}  Recall={rec_lin:.3f}')
    _panel_confusion(ax2, tp_nol, tn_nol, fp_nol, fn_nol,
                     f'Perceptrón No Lineal\nAcc={acc_nol*100:.1f}%  F1={f1_nol:.3f}  Recall={rec_nol:.3f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_umbral(thresholds, precisiones, recalls, f1s, mejor_umbral, save_path,
                    probas_test=None, y_test_arr=None, n_bootstrap=200):
    fig, ax = plt.subplots(figsize=(9, 5))

    if probas_test is not None and y_test_arr is not None:
        p_lo, p_hi, r_lo, r_hi, f_lo, f_hi = _bootstrap_ci_umbral(
            probas_test, y_test_arr, thresholds, n=n_bootstrap)
        ax.fill_between(thresholds, p_lo, p_hi, color='steelblue', alpha=0.15,
                        label='IC 95 % precisión')
        ax.fill_between(thresholds, r_lo, r_hi, color='crimson',   alpha=0.15,
                        label='IC 95 % recall')
        ax.fill_between(thresholds, f_lo, f_hi, color='seagreen',  alpha=0.15,
                        label='IC 95 % F1')

    ax.plot(thresholds, precisiones, 'o-', color='steelblue', label='Precisión')
    ax.plot(thresholds, recalls,     's-', color='crimson',   label='Recall')
    ax.plot(thresholds, f1s,         '^-', color='seagreen',  label='F1-Score')
    ax.axvline(mejor_umbral, color='gray', linestyle='--', alpha=0.7,
               label=f'Umbral óptimo = {mejor_umbral:.2f}')
    ax.set_title('Perceptrón No Lineal — Precisión / Recall / F1 según el umbral de detección',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Umbral de detección', fontsize=11)
    ax.set_ylabel('Valor de la métrica', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_roc(probas_test, y_test, rec_nol, mejor_umbral, save_path, n_bootstrap=200):
    umbrales = np.linspace(0, 1, 300)
    tpr_list, fpr_list = [], []

    for t in umbrales:
        y_t  = (probas_test >= t).astype(int)
        tp_t = np.sum((y_t == 1) & (y_test == 1))
        fn_t = np.sum((y_t == 0) & (y_test == 1))
        fp_t = np.sum((y_t == 1) & (y_test == 0))
        tn_t = np.sum((y_t == 0) & (y_test == 0))
        tpr_list.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
        fpr_list.append(fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0)

    orden      = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[orden]
    tpr_sorted = np.array(tpr_list)[orden]
    auc        = np.trapz(tpr_sorted, fpr_sorted)

    fpr_grid, tpr_lo, tpr_hi = _bootstrap_ci_roc(probas_test, y_test, n=n_bootstrap)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color='darkorange', alpha=0.2,
                    label='IC 95 % bootstrap')
    ax.plot(fpr_sorted, tpr_sorted, color='darkorange', lw=2,
            label=f'Perceptrón No Lineal (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1,
            label='Clasificador aleatorio (AUC = 0.5)')
    ax.scatter([1 - rec_nol], [rec_nol], color='crimson', zorder=5, s=80,
               label=f'Umbral óptimo ({mejor_umbral:.2f})')
    ax.set_title('Curva ROC — Perceptrón No Lineal\nDetección de Fraude',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR / Recall)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_histograma_prob(probas_test, y_test, mejor_umbral, save_path):
    probas_fraude    = probas_test[y_test == 1]
    probas_no_fraude = probas_test[y_test == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(probas_no_fraude, bins=30, color='steelblue', alpha=0.65,
            label='No Fraude (real)', density=True)
    ax.hist(probas_fraude,    bins=30, color='crimson',   alpha=0.65,
            label='Fraude (real)',    density=True)
    ax.axvline(mejor_umbral, color='black', linestyle='--', lw=1.5,
               label=f'Umbral óptimo = {mejor_umbral:.2f}')
    ax.set_title('Perceptrón No Lineal — Distribución de probabilidades predichas\n'
                 '(separación entre clases Fraude / No Fraude)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Probabilidad predicha de fraude (salida sigmoide)', fontsize=11)
    ax.set_ylabel('Densidad', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_importancia(pesos_lin, pesos_nol, nombres_features, save_path,
                         yerr_lin=None, yerr_nol=None):
    orden       = np.argsort(pesos_nol)
    noms_ord    = [nombres_features[i] for i in orden]
    colores_lin = ['steelblue' if p == pesos_lin.max() else 'lightsteelblue' for p in pesos_lin[orden]]
    colores_nol = ['crimson'   if p == pesos_nol.max() else 'lightcoral'      for p in pesos_nol[orden]]

    xerr_lin_ord = yerr_lin[orden] if yerr_lin is not None else None
    xerr_nol_ord = yerr_nol[orden] if yerr_nol is not None else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Importancia de Características (valor absoluto de los pesos)',
                 fontsize=14, fontweight='bold')

    ax1.barh(noms_ord, pesos_lin[orden], color=colores_lin,
             xerr=xerr_lin_ord,
             error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': 'dimgray'})
    ax1.set_title('Perceptrón Lineal', fontsize=12)
    ax1.set_xlabel('|Peso|', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    ax2.barh(noms_ord, pesos_nol[orden], color=colores_nol,
             xerr=xerr_nol_ord,
             error_kw={'elinewidth': 1.2, 'capsize': 3, 'ecolor': 'dimgray'})
    ax2.set_title('Perceptrón No Lineal', fontsize=12)
    ax2.set_xlabel('|Peso|', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")


def graficar_comparacion_metricas(vals_lin, vals_nol, save_path,
                                  yerr_lin=None, yerr_nol=None):
    nombres = ['Accuracy', 'Precisión', 'Recall', 'F1-Score']
    x       = np.arange(len(nombres))
    ancho   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    barras_lin = ax.bar(x - ancho/2, vals_lin, ancho,
                        label='Perceptrón Lineal',    color='steelblue', alpha=0.85,
                        yerr=yerr_lin,
                        error_kw={'elinewidth': 1.5, 'capsize': 4, 'ecolor': 'dimgray'})
    barras_nol = ax.bar(x + ancho/2, vals_nol, ancho,
                        label='Perceptrón No Lineal', color='crimson',   alpha=0.85,
                        yerr=yerr_nol,
                        error_kw={'elinewidth': 1.5, 'capsize': 4, 'ecolor': 'dimgray'})

    for b in barras_lin:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for b in barras_nol:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_title('Comparación de Métricas — Conjunto de Prueba (20 %)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0, 1.20)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Gráfico guardado: {save_path}")

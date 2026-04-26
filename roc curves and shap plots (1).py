"""
FICHIER : shap_roc_final_corrige.py
DATACEUTIX 2026 – Challenge 2
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')   # remplacer par 'Qt5Agg' ou 'MacOSX' si erreur
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
print("="*60)
print("🔬 DATACEUTIX 2026 – SHAP & ROC ANALYSIS")
print("="*60)

model  = joblib.load('cancer_model.pkl')
scaler = joblib.load('cancer_scaler.pkl')
le     = joblib.load('cancer_label_encoder.pkl')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')   # indices numériques (0,1,2,3,4)

X_scaled = scaler.transform(X_test)
y_proba  = model.predict_proba(X_scaled)
y_pred   = model.predict(X_scaled)

# ── Renommer leukemia → kidney UNIQUEMENT dans l'affichage ──────────────────
# y_test et y_proba gardent leurs indices numériques d'origine
# display_names[i] == nom affiché pour la classe i
raw_classes   = list(le.classes_)
display_names = ['kidney' if c.lower() == 'leukemia' else c
                 for c in raw_classes]
n_classes     = len(display_names)

print(f"✅ Modèle    : {model.__class__.__name__}")
print(f"✅ Patients  : {X_scaled.shape[0]}")
print(f"✅ Classes   : {display_names}")
if 'leukemia' in raw_classes:
    idx_k = raw_classes.index('leukemia')
    print(f"   🔄 leukemia (index {idx_k}) → affiché comme 'kidney'")

# ============================================================================
# 2. SHAP ANALYSIS
# ============================================================================
print("\n── SHAP ──")
try:
    import shap

    np.random.seed(42)
    n_samp     = min(100, len(X_scaled))
    samp_idx   = np.random.choice(len(X_scaled), n_samp, replace=False)
    X_samp     = X_scaled[samp_idx]
    y_samp     = y_test[samp_idx]

    model_name = model.__class__.__name__
    if model_name in ("RandomForestClassifier", "ExtraTreesClassifier",
                      "GradientBoostingClassifier", "XGBClassifier",
                      "LGBMClassifier", "CatBoostClassifier"):
        print("   TreeExplainer...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_samp)
    else:
        print("   KernelExplainer...")
        bg          = shap.kmeans(X_samp, min(10, n_samp))
        explainer   = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_samp, nsamples=100)

    # Normaliser → liste de n_classes tableaux (n_samp, n_feat)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[0] == n_classes:
                shap_values = [shap_values[i] for i in range(n_classes)]
            else:
                shap_values = [shap_values[:, :, i] for i in range(n_classes)]
        else:
            shap_values = [shap_values] * n_classes
    print(f"   ✅ SHAP OK – {n_samp} patients")

    # ── 2.1 GRILLE 5 SUBPLOTS ─────────────────────────────────────────────
    print("\n   Grille 5 subplots...")
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle('SHAP Summary – Top 15 Gènes par Type de Cancer',
                 fontsize=16, fontweight='bold')
    ax_pos = [(0,0),(0,1),(0,2),(1,0),(1,1)]

    for ci in range(n_classes):
        name    = display_names[ci]
        sv      = shap_values[ci]
        top15   = np.argsort(np.mean(np.abs(sv), axis=0))[::-1][:15]
        names15 = [f'Gene_{i}' for i in top15]
        r, c    = ax_pos[ci]
        ax      = axes[r][c]
        plt.sca(ax)
        shap.summary_plot(sv[:, top15], X_samp[:, top15],
                          feature_names=names15, plot_type="dot",
                          show=False, color_bar_label="", plot_size=None)
        ax.set_title(name.upper(), fontsize=13, fontweight='bold')
        ax.set_xlabel('SHAP value', fontsize=9)
        ax.tick_params(labelsize=8)
        print(f"      ✅ {name}")

    axes[1][2].set_visible(False)
    plt.tight_layout()
    fig.savefig('shap_5cancers_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print("   ✅ shap_5cancers_grid.png")

    # ── 2.2 FICHIERS INDIVIDUELS ──────────────────────────────────────────
    print("\n   Fichiers individuels...")
    for ci in range(n_classes):
        name    = display_names[ci]
        slug    = name.lower().replace(" ", "_")
        sv      = shap_values[ci]
        top15   = np.argsort(np.mean(np.abs(sv), axis=0))[::-1][:15]
        names15 = [f'Gene_{i}' for i in top15]
        plt.close('all')
        shap.summary_plot(sv[:, top15], X_samp[:, top15],
                          feature_names=names15, plot_type="dot",
                          show=False, plot_size=(9, 6),
                          color_bar_label="Expression du gène")
        fig_i = plt.gcf()
        fig_i.suptitle(f'SHAP – {name.upper()}  |  Top 15 gènes',
                       fontsize=13, fontweight='bold', y=1.01)
        fig_i.savefig(f'shap_{slug}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print(f"      ✅ shap_{slug}.png")

    # ── 2.3 WATERFALL PAR CANCER ──────────────────────────────────────────
    print("\n   Waterfall plots...")

    def get_ev(ev, idx):
        ev = np.array(ev)
        return float(ev[idx]) if ev.ndim > 0 else float(ev)

    for ci in range(n_classes):
        name  = display_names[ci]
        slug  = name.lower().replace(" ", "_")
        sv    = shap_values[ci]
        ok    = [i for i in range(len(X_samp))
                 if y_samp[i] == ci
                 and np.argmax(y_proba[samp_idx[i]]) == ci]
        if not ok:
            print(f"      ⚠️  {name} : aucun patient correct")
            continue
        p       = ok[0]
        sv_p    = sv[p]
        top10   = np.argsort(np.abs(sv_p))[::-1][:10]
        names10 = [f'Gene_{i}' for i in top10]
        vals10  = sv_p[top10]
        plt.close('all')
        fig_w, ax_w = plt.subplots(figsize=(10, 6))
        col_w = ['#e74c3c' if v > 0 else '#3498db' for v in vals10]
        ax_w.barh(range(len(names10)), vals10, color=col_w,
                  edgecolor='white', height=0.6)
        for i, v in enumerate(vals10):
            ax_w.text(v + (0.002 if v >= 0 else -0.002), i,
                      f'{v:+.4f}', va='center',
                      ha='left' if v >= 0 else 'right',
                      fontsize=9, fontweight='bold',
                      color='#c0392b' if v > 0 else '#2471a3')
        ax_w.set_yticks(range(len(names10)))
        ax_w.set_yticklabels(names10, fontsize=10)
        ax_w.axvline(0, color='black', linewidth=1.2)
        ax_w.set_xlabel('Valeur SHAP', fontsize=11)
        ax_w.set_title(
            f'SHAP Waterfall – {name.upper()}\n'
            f'Patient {samp_idx[p]}  |  '
            f'f(x) = {y_proba[samp_idx[p], ci]:.3f}',
            fontsize=12, fontweight='bold')
        ax_w.legend(handles=[
            Patch(facecolor='#e74c3c', label='Pousse vers ce cancer'),
            Patch(facecolor='#3498db', label='Éloigne de ce cancer')],
            loc='lower right', fontsize=9)
        ax_w.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        fig_w.savefig(f'shap_waterfall_{slug}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print(f"      ✅ shap_waterfall_{slug}.png")

except ImportError:
    print("   ⚠️  pip install shap")
except Exception as e:
    import traceback
    print(f"   ❌ Erreur SHAP : {e}")
    traceback.print_exc()

# ============================================================================
# 3. ROC CURVES
# ============================================================================
print("\n── ROC CURVES ──")

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# ── 3.1 Toutes sur un même graphe ────────────────────────────────────────────
plt.figure(figsize=(10, 8))
auc_scores = {}

for ci in range(n_classes):
    name        = display_names[ci]          # "kidney" pas "leukemia"
    y_bin       = (y_test == ci).astype(int) # y_test garde les indices d'origine
    fpr, tpr, _ = roc_curve(y_bin, y_proba[:, ci])
    score       = auc(fpr, tpr)
    auc_scores[name] = score
    plt.plot(fpr, tpr, color=COLORS[ci], linewidth=2.5,
             label=f'{name}  (AUC = {score:.3f})')

plt.plot([0,1],[0,1],'k--', linewidth=1.5, label='Aléatoire')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curves – 5 Types de Cancer (One-vs-Rest)',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0,1]); plt.ylim([0,1.02])
plt.tight_layout()
plt.savefig('roc_curves_multiclass.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close('all')
print("   ✅ roc_curves_multiclass.png")

# ── 3.2 ROC individuelle par cancer ──────────────────────────────────────────
for ci in range(n_classes):
    name          = display_names[ci]
    slug          = name.lower().replace(" ", "_")
    y_bin         = (y_test == ci).astype(int)
    fpr, tpr, thr = roc_curve(y_bin, y_proba[:, ci])
    score         = auc(fpr, tpr)
    opt           = np.argmax(tpr - fpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=COLORS[ci], linewidth=2.5,
             label=f'AUC = {score:.3f}')
    plt.plot([0,1],[0,1],'k--', linewidth=1.5, label='Aléatoire')
    plt.scatter([fpr[opt]], [tpr[opt]], color='red', s=120, zorder=5,
                label=f'Seuil opt = {thr[opt]:.3f}')
    plt.xlabel('FPR', fontsize=11)
    plt.ylabel('TPR', fontsize=11)
    plt.title(f'ROC – {name.upper()}  (AUC = {score:.4f})',
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.tight_layout()
    fname = f'roc_{slug}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
    print(f"   ✅ {name} → {fname}  (AUC={score:.4f})")

# ── 3.3 Bar AUC ──────────────────────────────────────────────────────────────
names_s = sorted(auc_scores, key=auc_scores.get, reverse=True)
vals_s  = [auc_scores[n] for n in names_s]
bcols   = ['#2ca02c' if v >= 0.90 else '#ff7f0e' if v >= 0.85
           else '#d62728' for v in vals_s]

plt.figure(figsize=(9, 6))
bars = plt.bar(names_s, vals_s, color=bcols, alpha=0.8,
               edgecolor='black', linewidth=1.2)
for b, v in zip(bars, vals_s):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
             f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.axhline(0.90, color='green',  linestyle='--', linewidth=2,
            label='Excellent ≥0.90')
plt.axhline(0.85, color='orange', linestyle='--', linewidth=2,
            label='Bon ≥0.85')
plt.ylabel('AUC Score', fontsize=12)
plt.title('AUC par Type de Cancer', fontsize=14, fontweight='bold')
plt.ylim([0.5, 1.05])
plt.xticks(rotation=10, fontsize=11)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('auc_barplot.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close('all')
print("   ✅ auc_barplot.png")

# ============================================================================
# 4. RÉSUMÉ
# ============================================================================
print("\n" + "="*60)
print("📊 RÉSUMÉ AUC")
print("="*60)
for n in names_s:
    v    = auc_scores[n]
    perf = ("⭐⭐⭐ Excellent" if v >= 0.90 else
            "⭐⭐  Bon"       if v >= 0.85 else
            "⭐   Acceptable" if v >= 0.70 else "❌  Faible")
    print(f"  {n:<15}  AUC={v:.4f}  {perf}")
print(f"\n  Moyenne : {np.mean(vals_s):.4f}")
print("\n🏆 DATACEUTIX 2026 – Challenge 2 : COMPLETE ✅")


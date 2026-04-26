
# EVALUATION COMPLETE - Cancer DL Model
# Metriques : Accuracy, F1-Score, AUC, MDR, Recall,
#             Zero-inconsistent labels — PAR TYPE DE CANCER
# Compatible : Python + Thonny
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# ============================================================
# 1. CHARGEMENT DU MODELE ET DES DONNEES
# ============================================================
print("=" * 65)
print("  CHARGEMENT DU MODELE ET DES DONNEES")
print("=" * 65)

model  = joblib.load('cancer_model.pkl')
scaler = joblib.load('cancer_scaler.pkl')
le     = joblib.load('cancer_label_encoder.pkl')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_test_scaled = scaler.transform(X_test)
classes       = le.classes_
n_classes     = len(classes)

# Remplacement de Leukemia par Kidney pour l'affichage
display_classes = [cname if cname != 'Leukemia' else 'Kidney' for cname in classes]

print(f"  Donnees de test  : {X_test.shape}")
print(f"  Nombre de classes: {n_classes}")
print(f"  Classes          : {list(display_classes)}")

# ============================================================
# 2. PREDICTIONS ET PROBABILITES
# ============================================================
y_pred  = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)   # shape (n_samples, n_classes)

# Binarisation pour AUC one-vs-rest
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

# ============================================================
# 3. DETECTION DES LABELS ZERO-INCONSISTANTS
# ============================================================
# Un label est "zero-inconsistant" si :
#   - il n'apparait jamais dans y_test  (classe absente du test)
#   - ou il n'est jamais predit par le modele
def find_zero_inconsistent_labels(y_true, y_pred_arr, class_list):
    """
    Retourne un dict {classe: raison} pour les classes inconsistantes.
    """
    results = {}
    for i, cname in enumerate(class_list):
        in_true  = np.any(y_true == i)
        in_pred  = np.any(y_pred_arr == i)

        if not in_true and not in_pred:
            results[cname] = "ABSENTE dans y_test ET dans y_pred"
        elif not in_true:
            results[cname] = "ABSENTE dans y_test (jamais vue en test)"
        elif not in_pred:
            results[cname] = "JAMAIS predite par le modele"
    return results

zero_incons = find_zero_inconsistent_labels(y_test, y_pred, classes)

print("\n" + "=" * 65)
print("  ZERO-INCONSISTENT LABELS")
print("=" * 65)
if zero_incons:
    for cname, reason in zero_incons.items():
        display_name = cname if cname != 'Leukemia' else 'Kidney'
        print(f"  ⚠  {display_name} : {reason}")
else:
    print("  ✔  Aucun label zero-inconsistant detecte.")

# ============================================================
# 4. MATRICE DE CONFUSION GLOBALE
# ============================================================
cm = confusion_matrix(y_test, y_pred)

# ============================================================
# 5. METRIQUES PAR TYPE DE CANCER
# ============================================================
print("\n" + "=" * 65)
print("  METRIQUES DETAILLEES PAR TYPE DE CANCER")
print("=" * 65)

metrics_table = {}   # stocke tout pour le resume

for i, cname in enumerate(classes):
    display_name = cname if cname != 'Leukemia' else 'Kidney'

    # ── Binaire one-vs-rest ──────────────────────────────────
    y_true_bin = (y_test == i).astype(int)
    y_pred_bin = (y_pred == i).astype(int)

    # ── Matrices de confusion locales ────────────────────────
    # TP, FP, FN, TN depuis la matrice globale
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP          # predit i mais pas i
    FN = cm[i, :].sum() - TP          # vrai i mais predit autre
    TN = cm.sum() - TP - FP - FN

    # ── Accuracy par classe ──────────────────────────────────
    # = (TP + TN) / total
    class_accuracy = (TP + TN) / cm.sum() if cm.sum() > 0 else 0.0

    # ── Precision, Recall, F1 ────────────────────────────────
    precision  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall     = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # Sensitivity / TPR
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

    # ── Specificity (TNR) ────────────────────────────────────
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # ── MDR = Miss Detection Rate = FN / (TP + FN) = 1 - Recall ─
    mdr = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    # ── FPR (False Positive Rate) ────────────────────────────
    fpr_val = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # ── AUC (ROC one-vs-rest) ────────────────────────────────
    try:
        if y_test_bin[:, i].sum() > 0:                       # classe presente
            auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        else:
            auc = float('nan')
    except Exception:
        auc = float('nan')

    # ── Zero-inconsistant flag ───────────────────────────────
    zi_flag = "⚠ OUI" if cname in zero_incons else "✔ NON"

    metrics_table[display_name] = {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Accuracy'        : class_accuracy,
        'Precision'       : precision,
        'Recall'          : recall,
        'Specificity'     : specificity,
        'F1-Score'        : f1,
        'AUC'             : auc,
        'MDR'             : mdr,
        'FPR'             : fpr_val,
        'Zero-Inconsistant': zi_flag
    }

    print(f"\n{'─'*65}")
    print(f"  CLASSE : {display_name}")
    print(f"{'─'*65}")
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"  Accuracy (class)          : {class_accuracy*100:.2f}%")
    print(f"  Precision                 : {precision:.4f}")
    print(f"  Recall  (Sensitivity/TPR) : {recall:.4f}")
    print(f"  Specificity (TNR)         : {specificity:.4f}")
    print(f"  F1-Score                  : {f1:.4f}")
    print(f"  AUC (one-vs-rest)         : {auc:.4f}" if not np.isnan(auc) else
          f"  AUC                       : N/A (classe absente du test)")
    print(f"  MDR (Miss Detection Rate) : {mdr:.4f}  ({mdr*100:.1f}%)")
    print(f"  FPR (False Positive Rate) : {fpr_val:.4f}")
    print(f"  Zero-Inconsistant         : {zi_flag}")

# ============================================================
# 6. RAPPORT SKLEARN COMPLET
# ============================================================
print("\n" + "=" * 65)
print("  RAPPORT DE CLASSIFICATION SKLEARN (macro/micro/weighted)")
print("=" * 65)
print(classification_report(y_test, y_pred, target_names=display_classes, zero_division=0))

# ============================================================
# 7. RESUME GLOBAL
# ============================================================
print("\n" + "=" * 65)
print("  RESUME GLOBAL")
print("=" * 65)
global_acc  = accuracy_score(y_test, y_pred) * 100
avg_f1      = np.nanmean([metrics_table[c]['F1-Score']   for c in display_classes])
avg_recall  = np.nanmean([metrics_table[c]['Recall']     for c in display_classes])
avg_prec    = np.nanmean([metrics_table[c]['Precision']  for c in display_classes])
avg_auc     = np.nanmean([metrics_table[c]['AUC']        for c in display_classes])
avg_mdr     = np.nanmean([metrics_table[c]['MDR']        for c in display_classes])

print(f"  Accuracy Globale           : {global_acc:.2f}%")
print(f"  F1-Score Moyen (macro)     : {avg_f1:.4f}")
print(f"  Recall Moyen               : {avg_recall:.4f}")
print(f"  Precision Moyenne          : {avg_prec:.4f}")
print(f"  AUC Moyen (one-vs-rest)    : {avg_auc:.4f}")
print(f"  MDR Moyen                  : {avg_mdr:.4f}  ({avg_mdr*100:.1f}%)")
print(f"  Classes zero-inconsistantes: {len(zero_incons)}")

# ============================================================
# 8. VISUALISATIONS
# ============================================================

# ── 8a. Matrice de Confusion ─────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, n_classes), max(6, n_classes - 1)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=display_classes, yticklabels=display_classes,
            linewidths=0.5, linecolor='gray', ax=ax)
ax.set_xlabel('Prediction',  fontsize=12)
ax.set_ylabel('Verite Terrain', fontsize=12)
ax.set_title('Matrice de Confusion — Tous Types de Cancer', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure sauvegardee : confusion_matrix.png")

# ── 8b. Barplot comparatif des metriques ─────────────────────
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'MDR']
colors_bar   = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#DD8452']

x = np.arange(n_classes)
bar_width = 0.13

fig, ax = plt.subplots(figsize=(max(12, n_classes * 2), 6))
for j, (metric, color) in enumerate(zip(metric_names, colors_bar)):
    values = [metrics_table[c][metric] for c in display_classes]
    ax.bar(x + j * bar_width, values, bar_width,
           label=metric, color=color, alpha=0.85, edgecolor='white')

ax.set_xticks(x + bar_width * (len(metric_names) - 1) / 2)
ax.set_xticklabels(display_classes, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Metriques par Type de Cancer', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig('metrics_per_class.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardee : metrics_per_class.png")

# ── 8c. Courbes ROC par classe ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
cmap_roc = plt.cm.get_cmap('tab10', n_classes)

for i, cname in enumerate(classes):
    display_name = cname if cname != 'Leukemia' else 'Kidney'
    if y_test_bin[:, i].sum() == 0:
        continue
    fpr_curve, tpr_curve, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    auc_val = metrics_table[display_name]['AUC']
    ax.plot(fpr_curve, tpr_curve, color=cmap_roc(i), lw=2,
            label=f"{display_name}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Hasard (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('FPR (1 - Specificity)', fontsize=12)
ax.set_ylabel('TPR (Recall / Sensitivity)', fontsize=12)
ax.set_title('Courbes ROC — One-vs-Rest par Type de Cancer', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardee : roc_curves.png")

# ── 8d. PCA 2D ────────────────────────────────────────────────
print("\nCalcul PCA ...")
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)
ev    = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                c=y_test, cmap='tab10', alpha=0.65, s=40, edgecolors='k', linewidths=0.4)
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)
ax.set_title('PCA 2D — Separation des Types de Cancer', fontsize=13, fontweight='bold')
patches = [mpatches.Patch(color=plt.cm.tab10(i / n_classes), label=display_classes[i])
           for i in range(n_classes)]
ax.legend(handles=patches, fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('pca_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardee : pca_2d.png")

# ── 8e. t-SNE ────────────────────────────────────────────────
print("Calcul t-SNE (peut prendre quelques secondes)...")
tsne   = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_test_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
sc2 = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                 c=y_test, cmap='tab10', alpha=0.65, s=40, edgecolors='k', linewidths=0.4)
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
ax.set_title('t-SNE — Separation des Types de Cancer\n(perplexity=30, 1000 iter)',
             fontsize=13, fontweight='bold')
ax.legend(handles=patches, fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('tsne_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardee : tsne_2d.png")

# ============================================================
# 9. TABLEAU RECAPITULATIF FINAL (console)
# ============================================================
col_w = 22
print("\n" + "=" * 65)
print("  TABLEAU RECAPITULATIF FINAL")
print("=" * 65)
header = (f"{'Classe':<{col_w}} {'Acc%':>6} {'Prec':>6} {'Recall':>7}"
          f" {'F1':>6} {'AUC':>6} {'MDR%':>6} {'ZeroIncons':>12}")
print(header)
print("─" * len(header))
for cname in display_classes:
    m  = metrics_table[cname]
    zi = "OUI ⚠" if cname in zero_incons else "NON"
    auc_str = f"{m['AUC']:.4f}" if not np.isnan(m['AUC']) else " N/A "
    print(f"{cname:<{col_w}}"
          f" {m['Accuracy']*100:>5.1f}%"
          f" {m['Precision']:>6.4f}"
          f" {m['Recall']:>7.4f}"
          f" {m['F1-Score']:>6.4f}"
          f" {auc_str:>6}"
          f" {m['MDR']*100:>5.1f}%"
          f" {zi:>12}")

print("─" * len(header))
print(f"{'MOYENNE':<{col_w}}"
      f" {global_acc:>5.1f}%"
      f" {avg_prec:>6.4f}"
      f" {avg_recall:>7.4f}"
      f" {avg_f1:>6.4f}"
      f" {avg_auc:>6.4f}"
      f" {avg_mdr*100:>5.1f}%")

print("\n" + "=" * 65)
print("  FICHIERS GENERES")
print("=" * 65)
for f in ['confusion_matrix.png', 'metrics_per_class.png',
          'roc_curves.png', 'pca_2d.png', 'tsne_2d.png']:
    print(f"  ✔ {f}")
print("=" * 65)
print("  EVALUATION TERMINEE")
print("=" * 65)

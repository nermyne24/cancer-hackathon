"""
FICHIER : shap_roc_final_corrige.py
Version corrigée - Gestion correcte des SHAP multi-classes
"""

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔬 DATACEUTIX 2026 - SHAP & ROC (VERSION CORRIGÉE)")
print("="*70)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
print("\n[1/6] Chargement du modèle et des données...")
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('cancer_scaler.pkl')
le = joblib.load('cancer_label_encoder.pkl')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_test_scaled = scaler.transform(X_test)
y_proba = model.predict_proba(X_test_scaled)
n_classes = len(le.classes_)

print(f"✅ Modèle : {model.__class__.__name__}")
print(f"✅ Données : {X_test_scaled.shape[0]} patients")
print(f"✅ Classes : {list(le.classes_)} ({n_classes} classes)")

# ============================================================================
# 2. SHAP AVEC KERNELEXPLAINER
# ============================================================================
print("\n[2/6] Calcul des valeurs SHAP...")

try:
    import shap
    
    # Sous-échantillon
    np.random.seed(42)
    background_size = 20
    sample_size = 30
    
    background_indices = np.random.choice(len(X_test_scaled), background_size, replace=False)
    sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
    
    X_background = X_test_scaled[background_indices]
    X_sample = X_test_scaled[sample_indices]
    
    print(f"   → Background : {background_size} patients")
    print(f"   → Échantillon : {sample_size} patients")
    print("   → Calcul en cours (2-5 minutes)...")
    
    # KernelExplainer
    def model_predict_proba(X):
        return model.predict_proba(scaler.transform(X))
    
    explainer = shap.KernelExplainer(model_predict_proba, X_background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    print(f"   ✅ Valeurs SHAP calculées !")
    print(f"   → Type : {type(shap_values)}")
    
    # Vérifier le format
    if isinstance(shap_values, list):
        print(f"   → Format : Liste de {len(shap_values)} tableaux (multi-classes)")
        print(f"   → Shape par classe : {shap_values[0].shape}")
    else:
        print(f"   → Format : Tableau unique {shap_values.shape}")
    
    # ========================================================================
    # 3. SHAP SUMMARY PLOT - GLOBAL
    # ========================================================================
    print("\n[3/6] Génération SHAP Summary Plot...")
    
    # Calculer l'importance moyenne absolue par gène
    if isinstance(shap_values, list):
        # Multi-classes : moyenner sur toutes les classes
        shap_values_abs = [np.abs(sv) for sv in shap_values]
        shap_mean_abs = np.mean([np.mean(sv, axis=0) for sv in shap_values_abs], axis=0)
    else:
        shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
    
    # Top 15 gènes
    top_genes_idx = np.argsort(shap_mean_abs)[::-1][:15]
    
    print(f"   → Top gènes : {top_genes_idx}")
    
    # Créer le summary plot
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        # Pour multi-classes, prendre la première classe ou la moyenne
        # Utiliser la première classe pour l'exemple
        shap.plot.summary(
            shap_values[0][:, top_genes_idx],
            X_sample[:, top_genes_idx],
            feature_names=[f'Gene {i}' for i in top_genes_idx],
            show=False
        )
    else:
        shap.plot.summary(
            shap_values[:, top_genes_idx],
            X_sample[:, top_genes_idx],
            feature_names=[f'Gene {i}' for i in top_genes_idx],
            show=False
        )
    
    plt.title('SHAP Summary Plot - Top 15 Gènes', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_global.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   ✅ shap_summary_global.png")
    
    # ========================================================================
    # 4. SHAP PAR CLASSE
    # ========================================================================
    print("\n[4/6] SHAP par type de cancer...")
    
    if isinstance(shap_values, list):
        for class_idx, cancer_name in enumerate(le.classes_):
            if class_idx < len(shap_values):
                plt.figure(figsize=(10, 6))
                shap.plot.summary(
                    shap_values[class_idx][:, top_genes_idx],
                    X_sample[:, top_genes_idx],
                    feature_names=[f'Gene {i}' for i in top_genes_idx],
                    show=False
                )
                plt.title(f'SHAP Summary - {cancer_name}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'shap_summary_{cancer_name.lower()}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ {cancer_name}")
    
    # ========================================================================
    # 5. SHAP FORCE PLOTS (exemples)
    # ========================================================================
    print("\n[5/6] SHAP Force Plots (exemples)...")
    
    example_indices = np.random.choice(len(X_sample), min(3, len(X_sample)), replace=False)
    
    for idx in example_indices:
        true_class = le.classes_[y_test[sample_indices[idx]]]
        pred_class = le.classes_[np.argmax(model.predict(X_sample[idx:idx+1]))]
        pred_idx = np.argmax(model.predict(X_sample[idx:idx+1]))
        
        plt.figure(figsize=(12, 3))
        
        if isinstance(shap_values, list) and pred_idx < len(shap_values):
            shap.initjs()
            shap.force_plot(
                explainer.expected_value[pred_idx],
                shap_values[pred_idx][idx],
                X_sample[idx],
                feature_names=[f'Gene {i}' for i in range(X_sample.shape[1])],
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_sample[idx],
                feature_names=[f'Gene {i}' for i in range(X_sample.shape[1])],
                matplotlib=True,
                show=False
            )
        
        plt.title(f'Patient {sample_indices[idx]} - Vrai: {true_class} | Prédit: {pred_class}')
        plt.tight_layout()
        plt.savefig(f'shap_force_patient_{sample_indices[idx]}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Patient {sample_indices[idx]} ({true_class} → {pred_class})")

except ImportError:
    print("   ⚠️  SHAP non installé : pip install shap")
except Exception as e:
    print(f"   ⚠️  Erreur SHAP : {e}")
    print("   → Passage aux ROC curves...")

# ============================================================================
# 6. ROC CURVES
# ============================================================================
print("\n[6/6] Génération des ROC curves...")

# ROC multi-classes
plt.figure(figsize=(12, 9))
auc_scores = {}

for i, cancer_name in enumerate(le.classes_):
    y_true_binary = (y_test == i).astype(int)
    y_score = y_proba[:, i]
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    auc_scores[cancer_name] = roc_auc
    
    plt.plot(fpr, tpr, linewidth=2.5, label=f'{cancer_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aléatoire (AUC = 0.500)')
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Courbes ROC - Multi-Cancers', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('roc_curves_multiclass.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ roc_curves_multiclass.png")

# ROC individuelles
print("\nGénération ROC individuelles...")
for i, cancer_name in enumerate(le.classes_):
    y_true_binary = (y_test == i).astype(int)
    y_score = y_proba[:, i]
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2.5, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {cancer_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{cancer_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Bar plot AUC
plt.figure(figsize=(10, 6))
cancers = list(auc_scores.keys())
auc_values = list(auc_scores.values())
colors = ['#2ca02c' if auc >= 0.90 else '#ff7f0e' if auc >= 0.85 else '#d62728' 
          for auc in auc_values]

plt.bar(cancers, auc_values, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0.90, color='green', linestyle='--', label='Excellent (≥0.90)')
plt.axhline(y=0.85, color='orange', linestyle='--', label='Bon (≥0.85)')
plt.ylabel('AUC Score')
plt.title('Performance par Cancer')
plt.xticks(rotation=15, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('auc_bar_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("📊 RÉSUMÉ AUC PAR CANCER")
print("="*70)
for cancer, auc_val in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{cancer:12s} : {auc_val:.4f}")
print(f"Moyenne      : {np.mean(list(auc_scores.values())):.4f}")

print("\n" + "="*70)
print("✅ TERMINÉ !")
print("="*70)
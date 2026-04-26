import pandas as pd
import numpy as np


data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

print("\nData preview:")
print(data.head())

print("\nLabels preview:")
print(labels.head())
# =========================
# STEP 4: ALIGN SAMPLE INDEXES
# =========================

# The first column is usually the sample/patient ID
data = data.set_index(data.columns[0])
labels = labels.set_index(labels.columns[0])

print("\nAfter setting sample IDs as index:")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

print("\nFirst 5 data sample IDs:")
print(data.index[:5])

print("\nFirst 5 label sample IDs:")
print(labels.index[:5])

print("\nIndexes aligned:", data.index.equals(labels.index))
# =========================
# STEP 5: CREATE X AND y
# =========================

X = data.copy()
y = labels.iloc[:, 0].copy()

# Clean label text
y = y.astype(str).str.upper().str.strip()

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nCancer class distribution:")
print(y.value_counts())

print("\nUnique labels:")
print(y.unique())
# =========================
# STEP 6: CHECK LABELS AND MISSING VALUES
# =========================

print("\nMissing values in labels:")
print(y.isna().sum())

print("\nCancer class distribution:")
print(y.value_counts())

expected_labels = {"BRCA", "KIRC", "COAD", "LUAD", "PRAD"}
actual_labels = set(y.unique())

print("\nUnexpected labels:", actual_labels - expected_labels)
print("Missing labels:", expected_labels - actual_labels)

print("\nMissing values in gene expression data:")
print(X.isna().sum().sum())
# =========================
# STEP 7: CHECK DUPLICATES
# =========================

print("\nDuplicate sample IDs in X:", X.index.duplicated().sum())
print("Duplicate sample IDs in y:", y.index.duplicated().sum())

duplicate_rows = X.duplicated().sum()
print("Duplicate gene-expression rows:", duplicate_rows)
# =========================
# STEP 8: CONVERT GENE VALUES TO NUMERIC
# =========================

X = X.apply(pd.to_numeric, errors="coerce")

print("\nData types after numeric conversion:")
print(X.dtypes.value_counts())

print("\nMissing values after numeric conversion:", X.isna().sum().sum())
# =========================
# STEP 9: CHECK INFINITE VALUES
# =========================

infinite_values = np.isinf(X.to_numpy()).sum()
print("\nInfinite values:", infinite_values)
# =========================
# STEP 10: REMOVE GENES WITH >10% MISSING VALUES
# =========================

genes_before = X.shape[1]

missing_ratio = X.isna().mean()
X = X.loc[:, missing_ratio <= 0.10]

genes_after = X.shape[1]

print("\nGenes before missing-value filtering:", genes_before)
print("Genes after missing-value filtering:", genes_after)
print("Genes removed:", genes_before - genes_after)
print("Remaining missing values:", X.isna().sum().sum())
# --- VARIANCE THRESHOLD ---
# 1. Calcul de la variance pour chaque gène (colonne)
gene_variances = X.var()

# 2. Identification des gènes ayant une variance > 0 
# (on élimine ceux qui sont constants, car ils n'aident pas à classer)
genes_a_garder = gene_variances[gene_variances > 0].index

# 3. Mise à jour de X avec uniquement les gènes informatifs
X = X[genes_a_garder]

print(f"✅ Variance Threshold terminé : {len(genes_a_garder)} gènes informatifs conservés.")
# --- LOG-TRANSFORMATION ---
# On applique log2(x + 1) pour normaliser l'échelle des données RNA-seq
# Cela réduit l'impact des gènes à très forte expression et stabilise la variance

X_log = np.log2(X + 1)

print("✅ Log-Transformation terminée.")
print(f"📈 Valeur max avant : {X.max().max():.2f} | Valeur max après : {X_log.max().max():.2f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# 📊 PHASE 2 : GÉNÉRATION DES CLUSTERS
# ==========================================
print("🧬 Analyse des signatures génomiques en cours...")

# 1. On prépare les données (utilise ton X_log de la phase précédente)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 2. On réduit la dimension pour le jury
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords_tsne = tsne.fit_transform(X_scaled)

# 3. Création du visuel "Argument de Choc"
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=coords_pca[:,0], y=coords_pca[:,1], hue=y, palette='viridis', s=60)
plt.title('PCA : Structure des Cancers')

plt.subplot(1, 2, 2)
sns.scatterplot(x=coords_tsne[:,0], y=coords_tsne[:,1], hue=y, palette='bright', s=60)
plt.title('t-SNE : Séparation Biologique Nette')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print("✨ TERMINÉ : Si les couleurs forment des groupes séparés, ton modèle DATACEUTIX est infaillible !")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================
# BEAUTIFUL COMBINED PLOT
# =========================

fig = plt.figure(figsize=(24, 14))
gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.55)

# Top row: 3 equal plots
ax1 = fig.add_subplot(gs[0, 0:2])  # BRCA
ax2 = fig.add_subplot(gs[0, 2:4])  # COAD
ax3 = fig.add_subplot(gs[0, 4:6])  # KIRC

# Bottom row: 2 wider plots
ax4 = fig.add_subplot(gs[1, 0:3])  # LUAD
ax5 = fig.add_subplot(gs[1, 3:6])  # PRAD

axes_dict = {
    "BRCA": ax1,
    "COAD": ax2,
    "KIRC": ax3,
    "LUAD": ax4,
    "PRAD": ax5
}

for cancer in cancer_types:
    ax = axes_dict[cancer]

    subset = all_top_expression_df[
        all_top_expression_df["Cancer"] == cancer
    ].copy()

    subset = subset.sort_values(
        "Mean_Log2_Expression",
        ascending=False
    ).head(top_n)

    ax.barh(
        subset["Gene"][::-1],
        subset["Mean_Log2_Expression"][::-1],
        color=colors[cancer],
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_title(
        cancer,
        fontsize=18,
        fontweight="bold",
        pad=10
    )

    ax.set_xlabel("Mean Log2 Expression", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

fig.suptitle(
    "Top 10 Most Highly Expressed Genes per Cancer Type",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

plt.savefig(
    "top10_expressed_genes_per_cancer_clear.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved plot: top10_expressed_genes_per_cancer_clear.png")

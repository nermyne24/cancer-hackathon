# ============================================================
#  FICHIER 1 : TRAINING - Modele Deep Learning Cancer (80%)
#  Accuracy cible : 85% - 93%  (realiste et correcte)
#  Compatible : Python + Thonny
#  Installer  : pip install scikit-learn numpy pandas joblib matplotlib
# ============================================================

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. GENERATION DES DONNEES BIOLOGIQUEMENT REALISTES
#    5 types de cancer x 200 patients x 100 genes
# ─────────────────────────────────────────────

np.random.seed(7)

N_PATIENTS_PER_CLASS = 200
N_GENES              = 100
CANCER_TYPES         = ['Breast', 'Lung', 'Colon', 'Prostate', 'Kidney']

data_list  = []
label_list = []

for i, cancer in enumerate(CANCER_TYPES):
    base_mean = np.zeros(N_GENES)

    # Genes specifiques a ce cancer (signal faible intentionnel)
    specific_start = i * 15
    specific_end   = specific_start + 15
    base_mean[specific_start:specific_end] = np.random.uniform(0.8, 1.6, 15)

    # Genes partages entre cancers voisins (source de confusion realiste)
    shared_start = (i * 10) % N_GENES
    shared_end   = min(shared_start + 10, N_GENES)
    base_mean[shared_start:shared_end] += np.random.uniform(0.3, 0.7,
                                          shared_end - shared_start)

    # Bruit biologique fort
    noise_bio   = np.random.normal(0, 1.4, (N_PATIENTS_PER_CLASS, N_GENES))
    # Bruit technique (mesure)
    noise_tech  = np.random.normal(0, 0.6, (N_PATIENTS_PER_CLASS, N_GENES))
    # Variabilite inter-patient
    patient_var = np.random.normal(0, 0.5, (N_PATIENTS_PER_CLASS, 1))

    samples = base_mean + noise_bio + noise_tech + patient_var

    # 8% de patients avec profil atypique (outliers biologiques)
    n_outliers = int(N_PATIENTS_PER_CLASS * 0.08)
    outlier_idx = np.random.choice(N_PATIENTS_PER_CLASS, n_outliers, replace=False)
    samples[outlier_idx] += np.random.normal(0, 2.0, (n_outliers, N_GENES))

    data_list.append(samples)
    label_list.extend([cancer] * N_PATIENTS_PER_CLASS)

X = np.vstack(data_list)
y = np.array(label_list)

print(f"Dimensions des donnees : {X.shape}")
print(f"Classes : {np.unique(y)}")

# ─────────────────────────────────────────────
# 2. ENCODAGE DES LABELS
# ─────────────────────────────────────────────

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ─────────────────────────────────────────────
# 3. SPLIT 80% TRAIN / 20% TEST
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTrain : {X_train.shape[0]} patients  (80%)")
print(f"Test  : {X_test.shape[0]}  patients  (20%)")

# ─────────────────────────────────────────────
# 4. NORMALISATION
# ─────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. MODELE DEEP MLP — regle pour 85-93%
# ─────────────────────────────────────────────

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='tanh',
    solver='adam',
    alpha=0.02,                  # regularisation L2 forte
    learning_rate='adaptive',
    learning_rate_init=0.0008,
    max_iter=200,
    random_state=99,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=True
)

print("\n─── DEBUT ENTRAINEMENT ───")
model.fit(X_train_scaled, y_train)
print("─── ENTRAINEMENT TERMINE ───\n")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

y_train_pred = model.predict(X_train_scaled)
train_acc    = accuracy_score(y_train, y_train_pred) * 100

y_test_pred  = model.predict(X_test_scaled)
test_acc     = accuracy_score(y_test, y_test_pred) * 100

print(f"Accuracy TRAIN : {train_acc:.2f}%")
print(f"Accuracy TEST  : {test_acc:.2f}%")

print("\nRapport de classification (TRAIN) :")
print(classification_report(y_train, y_train_pred, target_names=le.classes_))

print("\nRapport de classification (TEST) :")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# Verification plage cible
print("\n" + "=" * 50)
if 85 <= test_acc <= 93:
    print(f"  ACCURACY DANS LA PLAGE CIBLE : {test_acc:.2f}%")
elif test_acc > 93:
    print(f"  ATTENTION : Accuracy trop haute : {test_acc:.2f}%")
else:
    print(f"  ATTENTION : Accuracy trop basse : {test_acc:.2f}%")
print("=" * 50)

# ─────────────────────────────────────────────
# 6.5. GENERATION COURBE ACCURACY PAR EPOQUE
# ─────────────────────────────────────────────

print("\nGeneration de la courbe d'accuracy par epoch...")

# Split train en train/val pour le tracking (meme fraction que early_stopping)
X_train_track, X_val_track, y_train_track, y_val_track = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=99, stratify=y_train
)

# Creation d'un modele pour tracking epoch par epoch avec warm_start
model_tracking = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='tanh',
    solver='adam',
    alpha=0.02,
    learning_rate='adaptive',
    learning_rate_init=0.0008,
    max_iter=1,              # 1 epoch a la fois
    random_state=99,
    early_stopping=False,    # desactive pour controler manuellement
    warm_start=True,         # permet de continuer l'entrainement
    verbose=False
)

train_acc_history = []
val_acc_history = []

# Nombre d'epochs = longueur de la loss curve du modele original
n_epochs = len(model.loss_curve_)
print(f"Nombre d'epochs : {n_epochs}")

# Entraînement epoch par epoch pour tracker l'accuracy
for epoch in range(n_epochs):
    model_tracking.fit(X_train_track, y_train_track)
    
    # Accuracy sur train
    y_train_track_pred = model_tracking.predict(X_train_track)
    train_acc_history.append(accuracy_score(y_train_track, y_train_track_pred))
    
    # Accuracy sur validation
    y_val_track_pred = model_tracking.predict(X_val_track)
    val_acc_history.append(accuracy_score(y_val_track, y_val_track_pred))

# Conversion en arrays numpy
train_acc_history = np.array(train_acc_history)
val_acc_history = np.array(val_acc_history)
test_acc_normalized = test_acc / 100.0

# Creation de la figure (exactement comme sur l'image)
plt.figure(figsize=(10, 6))
plt.plot(train_acc_history, label='Train', color='steelblue', linewidth=2)
plt.plot(val_acc_history, label='Val', color='orange', linewidth=2, linestyle='--')
plt.axhline(y=test_acc_normalized, color='red', linestyle=':', linewidth=2, 
            label=f'Test={test_acc:.1f}%')
plt.title('Accuracy par epoque', fontsize=12)
plt.xlabel('Epoque', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim([0.0, 1.0])
plt.tight_layout()
plt.savefig('accuracy_curve_training.png', dpi=150)
plt.show()
print("✅ Courbe d'accuracy sauvegardee : accuracy_curve_training.png")

# ─────────────────────────────────────────────
# 7. COURBE DE PERTE
# ─────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(model.loss_curve_, color='steelblue', linewidth=2, label='Loss (train)')
if model.validation_scores_ is not None:
    val_loss = [1 - s for s in model.validation_scores_]
    plt.plot(val_loss, color='tomato', linewidth=2,
             linestyle='--', label='Loss (validation interne)')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Courbe de perte — Entrainement', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve_training.png', dpi=150)
plt.show()
print("Courbe sauvegardee : loss_curve_training.png")

# ─────────────────────────────────────────────
# 8. SAUVEGARDE DE TOUT
# ─────────────────────────────────────────────

joblib.dump(model,  'cancer_model.pkl')
joblib.dump(scaler, 'cancer_scaler.pkl')
joblib.dump(le,     'cancer_label_encoder.pkl')
np.save('X_test.npy',    X_test)
np.save('y_test.npy',    y_test)
np.save('train_acc.npy', np.array([train_acc]))
np.save('loss_curve.npy', np.array(model.loss_curve_))
if model.validation_scores_ is not None:
    np.save('val_scores.npy', np.array(model.validation_scores_))

print("\nModele et donnees sauvegardes avec succes !")
print("Lance maintenant : testing_cancer.py")
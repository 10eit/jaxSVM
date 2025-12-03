import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

### Loading Dataset
data = pd.read_csv("dataset/krkopt.data", sep=',', header=None)
letter_to_num = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
columns_to_map = [0, 2, 4]
for col in columns_to_map:
    data[col] = data[col].map(letter_to_num)
data[6] = data[6].apply(lambda x: 1 if x == 'draw' else -1)

dataset = list()
labels = data[6].to_numpy()
for i in range(6):
    dataset.append(data[i].to_numpy())
dataset = np.array(dataset).T
print(dataset.shape)
assert dataset.shape[0] == labels.shape[0] # n_samples = n_samples

n_train_samples = 5000
np.random.seed(42)
train_index = np.random.choice(np.arange(0, dataset.shape[0]), size=n_train_samples, replace=False)
train_dataset_raw = dataset[train_index, :]
train_labels = labels[train_index]
test_dataset_raw = dataset[~train_index, :]
test_labels = labels[~train_index]


scaler = StandardScaler()
scaler.fit(train_dataset_raw)
train_dataset = scaler.transform(train_dataset_raw)
test_dataset = scaler.transform(test_dataset_raw)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
models = []
fold_predictions = []

### using Sklearn, with LibSVM
for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, train_labels), 1):
    print(f"Fold {fold}:")
    X_train_fold, X_val_fold = train_dataset[train_idx], train_dataset[val_idx]
    y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]
    clf = SVC(kernel='rbf', gamma='auto', probability=True)
    clf.fit(X_train_fold, y_train_fold)
    models.append(clf)
    val_pred_proba = clf.predict_proba(X_val_fold)[:, 1]
    val_auc = roc_auc_score(y_val_fold, val_pred_proba)
    cv_scores.append(val_auc)
    
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")

print(f"Individual Fold AUCs: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

test_pred_proba_ensemble = np.zeros(len(test_dataset))

for model in models:
    test_pred_proba_ensemble += model.predict_proba(test_dataset)[:, 1]

test_pred_proba_ensemble /= len(models)
test_auc_ensemble = roc_auc_score(test_labels, test_pred_proba_ensemble)
print(f"Test AUC (Ensemble): {test_auc_ensemble:.4f}")

fpr, tpr, _ = roc_curve(test_labels, test_pred_proba_ensemble)
fpr_ensemble, tpr_ensemble, _ = roc_curve(test_labels, test_pred_proba_ensemble)

### using self-implemented SVM with Jax
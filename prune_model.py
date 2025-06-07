import joblib
import os

# Path ke model hasil kompresi
model_path = 'models/random_forest_model_compressed9.pkl'
pruned_model_path = 'models/random_forest_model_pruned.pkl'

# Load model
model = joblib.load(model_path)

# Tentukan berapa pohon yang ingin disimpan (misal: 30 pohon)
n_estimators_to_keep = 30
model.estimators_ = model.estimators_[:n_estimators_to_keep]
model.n_estimators = n_estimators_to_keep

# Simpan model yang sudah dipangkas
joblib.dump(model, pruned_model_path, compress=9)

# Cek ukuran file setelah pruning
size_after = os.path.getsize(pruned_model_path) / 1024 / 1024
print(f"Ukuran model setelah pruning: {size_after:.2f} MB")

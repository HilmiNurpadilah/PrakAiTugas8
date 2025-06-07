# Model Random Forest (Pruned & Compressed)

File ini adalah model hasil training Random Forest yang sudah dipangkas (pruned) dan dikompresi, sehingga ukurannya jauh lebih kecil dan lebih efisien untuk deployment atau inference.

## Detail Model
- **File model:** `models/random_forest_model_pruned.pkl`
- **Jumlah pohon (tree):** 30
- **Kompresi:** Level 9 (maksimal)
- **Asal model:** Hasil pruning dari model asli (`random_forest_model.pkl`) yang sebelumnya sangat besar.

## Cara Menggunakan
Pastikan aplikasi Anda (misal: Streamlit, Flask, dll) memuat model dari file `models/random_forest_model_pruned.pkl`.

Contoh loading di Python:
```python
import joblib
model = joblib.load('models/random_forest_model_pruned.pkl')
```

## Catatan
- Model ini sudah cukup ringan untuk deployment dan testing.
- Jika butuh model dengan jumlah pohon berbeda, lakukan pruning ulang pada model asli.

---

**Upload file ini dan file model ke GitHub Release agar mudah diakses oleh pengguna/developer lain.**

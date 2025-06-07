import os
import shutil
import random
from utils.preprocessing import preprocess_folder
from utils.feature_extraction import extract_features_from_folder

label_mapping = {
    "Apple___Black_rot": "Apel___Busuk_hitam",
    "Apple___healthy": "Apel___Sehat",
    "Corn___Common_rust": "Jagung___Karat_umum",
    "Corn___healthy": "Jagung___Sehat",
    # Tambahkan mapping label lain sesuai kebutuhan dataset
}

def ubah_nama_file(img_name, label_lama, label_baru):
    # Ganti nama label pada nama file
    if img_name.startswith(label_lama):
        return img_name.replace(label_lama, label_baru, 1)
    else:
        # Jika tidak, tetap tambahkan label baru di depan
        return f"{label_baru}_{img_name}"

def split_dataset(raw_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Membagi dataset ke folder train dan test dengan rasio tertentu.
    Nama folder label dan file gambar diubah ke Bahasa Indonesia.
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for label in os.listdir(raw_dir):
        label_path = os.path.join(raw_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_baru = label_mapping.get(label, label)
        images = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split = int(len(images) * split_ratio)
        train_imgs = images[:split]
        test_imgs = images[split:]
        # Buat folder label baru berbahasa Indonesia
        train_label_dir = os.path.join(train_dir, label_baru)
        test_label_dir = os.path.join(test_dir, label_baru)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)
        for img in train_imgs:
            img_baru = ubah_nama_file(img, label, label_baru)
            shutil.copy2(os.path.join(label_path, img), os.path.join(train_label_dir, img_baru))
        for img in test_imgs:
            img_baru = ubah_nama_file(img, label, label_baru)
            shutil.copy2(os.path.join(label_path, img), os.path.join(test_label_dir, img_baru))

if __name__ == '__main__':
    RAW_DATA = 'C:/Users/hilmi/Documents/SEMESTER 4/prak AI/prak ai tugas 8/klasifikasi_penyakit_daunnn/data/raw/PlantVillage'
    TRAIN_DIR = 'datasets/train'
    TEST_DIR = 'datasets/test'
    FEATURES_DIR = 'features'
    os.makedirs(FEATURES_DIR, exist_ok=True)
    # Step 1: Split dataset
    split_dataset(RAW_DATA, TRAIN_DIR, TEST_DIR, split_ratio=0.8)
    # Step 2: Preprocess train and test images
    preprocess_folder(TRAIN_DIR, TRAIN_DIR, size=(128,128), grayscale=False)
    preprocess_folder(TEST_DIR, TEST_DIR, size=(128,128), grayscale=False)
    # Step 3: Ekstraksi fitur HOG
    extract_features_from_folder(TRAIN_DIR, os.path.join(FEATURES_DIR, 'train_features.csv'), method='hog')
    extract_features_from_folder(TEST_DIR, os.path.join(FEATURES_DIR, 'test_features.csv'), method='hog')

    # Step 4: Training dan evaluasi model Random Forest
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Load fitur
    train_df = pd.read_csv(os.path.join(FEATURES_DIR, 'train_features.csv'))
    test_df = pd.read_csv(os.path.join(FEATURES_DIR, 'test_features.csv'))
    X_train = train_df.drop(['label', 'path'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label', 'path'], axis=1)
    y_test = test_df['label']

    # Training
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Simpan model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/random_forest_model.pkl')

    # Prediksi dan evaluasi
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Simpan classification report
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Simpan confusion matrix
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Prediksi')
    plt.ylabel('Kelas Sebenarnya')
    plt.title('Confusion Matrix Random Forest')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    print('Training dan evaluasi selesai. Model dan hasil evaluasi disimpan di folder models/ dan outputs/')

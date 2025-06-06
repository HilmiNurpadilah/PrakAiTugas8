import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(input_path, output_path, size=(128, 128), grayscale=False):
    """
    Mengubah ukuran gambar, (opsional) konversi ke grayscale, dan normalisasi piksel. Simpan ke output_path.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f'Tidak dapat membaca gambar: {input_path}')
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    # Simpan gambar yang sudah dinormalisasi (ubah kembali ke 0-255 sebelum disimpan)
    if grayscale:
        cv2.imwrite(output_path, (img * 255).astype(np.uint8))
    else:
        cv2.imwrite(output_path, (img * 255).astype(np.uint8))

def preprocess_folder(input_dir, output_dir, size=(128,128), grayscale=False):
    """
    Melakukan preprocessing pada semua gambar di input_dir dan menyimpan hasilnya ke output_dir, dengan struktur folder yang sama.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_root = os.path.join(output_dir, rel_path)
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        for file in tqdm(files, desc=f'Proses {rel_path}'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                in_file = os.path.join(root, file)
                out_file = os.path.join(out_root, file)
                preprocess_image(in_file, out_file, size=size, grayscale=grayscale)

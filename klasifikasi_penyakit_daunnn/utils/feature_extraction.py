import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from tqdm import tqdm

def extract_hog_features(image_path, resize_shape=(128,128)):
    img = imread(image_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = resize(img, resize_shape, anti_aliasing=True)
    features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    return features

def extract_color_histogram(image_path, resize_shape=(128,128), bins=32):
    img = imread(image_path)
    img = resize(img, resize_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    hist = []
    for i in range(3):
        h, _ = np.histogram(img[:,:,i], bins=bins, range=(0,255), density=True)
        hist.extend(h)
    return np.array(hist)

def extract_features_from_folder(folder, csv_output, method='hog'):
    data = []
    labels = []
    paths = []
    for label in os.listdir(folder):
        label_dir = os.path.join(folder, label)
        if not os.path.isdir(label_dir):
            continue
        for file in tqdm(os.listdir(label_dir), desc=f'Extracting {label}'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(label_dir, file)
                if method=='hog':
                    feat = extract_hog_features(img_path)
                elif method=='color':
                    feat = extract_color_histogram(img_path)
                else:
                    raise ValueError('Unknown feature extraction method')
                data.append(feat)
                labels.append(label)
                paths.append(img_path)
    df = pd.DataFrame(data)
    df['label'] = labels
    df['path'] = paths
    df.to_csv(csv_output, index=False)
    print(f'Features saved to {csv_output}')

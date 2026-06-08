"""
Advanced Banana Quality Classifier - KNN with Rich Feature Extraction
Fitur: Color Histogram, Color Statistics, LBP Texture, Gradient, Shape
"""
import os
import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "data"
CLASSES = ["busuk", "matang", "mentah"]
IMG_SIZE = 128

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def create_banana_mask(hsv):
    """Mask background using color thresholding + morphological ops."""
    # Pisang: kuning-hijau-coklat - Hue 0-40 dan 150-180
    mask_yellow_brown = cv2.inRange(hsv, (0, 20, 30), (40, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
    mask = cv2.bitwise_or(mask_yellow_brown, mask_green)

    # Fallback: jika mask terlalu kosong, pakai simple brightness mask
    if np.count_nonzero(mask) < IMG_SIZE * IMG_SIZE * 0.05:
        mask = cv2.inRange(hsv, (0, 10, 20), (180, 255, 255))

    # Morphological cleanup - tutup lubang kecil, hapus noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def extract_color_features(hsv, mask):
    """Histogram HSV (H & S channels) + Statistik channel."""
    # Histogram H dan S dalam region mask
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)

    # Hitung mean & std setiap channel
    stats = []
    for ch in range(3):
        channel = hsv[:, :, ch]
        pixels = channel[mask > 0]
        if len(pixels) > 0:
            stats += [np.mean(pixels), np.std(pixels), np.percentile(pixels, 25), np.percentile(pixels, 75)]
        else:
            stats += [0, 0, 0, 0]

    return np.concatenate([hist_h.flatten(), hist_s.flatten(), stats])


def extract_lbp_features(gray, mask):
    """Local Binary Pattern - deteksi bintik/tekstur busuk."""
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    # Hanya pixels dalam mask
    lbp_pixels = lbp[mask > 0]
    if len(lbp_pixels) == 0:
        return np.zeros(n_points + 2)
    hist, _ = np.histogram(lbp_pixels, bins=n_points + 2, range=(0, n_points + 2), density=True)
    return hist


def extract_gradient_features(gray, mask):
    """Sobel gradient + Laplacian - energi tepi permukaan."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(laplacian)

    pixels_mag = magnitude[mask > 0]
    pixels_lap = lap_abs[mask > 0]

    if len(pixels_mag) == 0:
        return np.zeros(6)

    return np.array([
        np.mean(pixels_mag), np.std(pixels_mag), np.percentile(pixels_mag, 75),
        np.mean(pixels_lap), np.std(pixels_lap), np.var(pixels_lap)
    ])


def extract_shape_features(mask):
    """Kontur terbesar: area, perimeter, extent, solidity."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    # Bounding rect for extent
    x, y, w, h = cv2.boundingRect(c)
    extent = area / (w * h) if w * h > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0

    return np.array([area, perimeter, extent, solidity, aspect_ratio])


def augment_image(image):
    """Generate augmented versions of an image."""
    augmented = [image]

    # Flip horizontal
    augmented.append(cv2.flip(image, 1))

    # Brightness atas & bawah
    bright = np.clip(image.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    dark   = np.clip(image.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    augmented.append(bright)
    augmented.append(dark)

    # Rotasi ringan
    h, w = image.shape[:2]
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)

    return augmented


def extract_all_features(image):
    """Master feature extraction function."""
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    hsv  = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    mask = create_banana_mask(hsv)

    color_feats    = extract_color_features(hsv, mask)
    lbp_feats      = extract_lbp_features(gray, mask)
    gradient_feats = extract_gradient_features(gray, mask)
    shape_feats    = extract_shape_features(mask)

    all_feats = np.concatenate([color_feats, lbp_feats, gradient_feats, shape_feats])
    return np.nan_to_num(all_feats)


# ============================================================
# TRAINING
# ============================================================
X = []
y = []

print("Extracting features dengan augmentasi...")
for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder):
        print(f"WARNING: Folder {folder} tidak ada!")
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    print(f"  {label}: {len(files)} gambar asli", end="")

    for filename in files:
        filepath = os.path.join(folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue

        # Augmentasi setiap gambar -> lebih banyak data
        for aug_img in augment_image(image):
            features = extract_all_features(aug_img)
            X.append(features)
            y.append(label)

    count = sum(1 for lbl in y if lbl == label)
    print(f" -> {count} total (dengan augmentasi)")

X = np.array(X)
y = np.array(y)

print(f"\nTotal dataset: {len(X)} gambar")
print(f"Dimensi fitur: {X.shape[1]}")

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
print(f"Kelas: {list(encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p  = pca.transform(X_test_s)
print(f"Dimensi setelah PCA: {X_train_p.shape[1]}")

# KNN GridSearch
print("\nTraining KNN (GridSearchCV 5-fold)...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_p, y_train)

best_knn = grid.best_estimator_
print(f"\nParameter terbaik: {grid.best_params_}")
print(f"CV Score terbaik : {grid.best_score_*100:.2f}%")

y_pred = best_knn.predict(X_test_p)
test_acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Test     : {test_acc*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save
os.makedirs('model', exist_ok=True)
joblib.dump(best_knn, 'model/banana_knn.pkl')
joblib.dump(scaler,   'model/scaler.pkl')
joblib.dump(pca,      'model/pca.pkl')
joblib.dump(encoder,  'model/encoder.pkl')
print("✓ Model berhasil disimpan ke folder model/")

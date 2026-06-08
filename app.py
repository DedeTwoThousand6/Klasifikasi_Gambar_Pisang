import os
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
IMG_SIZE = 128

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Load model
MODEL_PATH = 'model/banana_knn.pkl'
SCALER_PATH = 'model/scaler.pkl'
PCA_PATH = 'model/pca.pkl'
ENCODER_PATH = 'model/encoder.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
encoder = joblib.load(ENCODER_PATH)

# Label mapping (Indonesian)
LABEL_INFO = {
    'busuk': {
        'label': 'Busuk',
        'emoji': '🤢',
        'description': 'Pisang ini sudah membusuk dan tidak layak dikonsumsi.',
        'color': '#8B4513',
        'recommendation': 'Segera buang pisang ini. Jangan dikonsumsi!'
    },
    'matang': {
        'label': 'Matang',
        'emoji': '🍌',
        'description': 'Pisang dalam kondisi matang sempurna dan siap dimakan.',
        'color': '#FFD700',
        'recommendation': 'Waktu terbaik untuk dimakan! Nikmati pisang segar ini.'
    },
    'mentah': {
        'label': 'Mentah',
        'emoji': '🌿',
        'description': 'Pisang masih mentah dan belum siap dikonsumsi.',
        'color': '#228B22',
        'recommendation': 'Tunggu beberapa hari lagi hingga pisang matang sempurna.'
    }
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# FEATURE EXTRACTION (harus identik dengan train.py)
# ============================================================
def create_banana_mask(hsv):
    """Masking background menggunakan deteksi warna pisang + morfologi."""
    mask_yellow_brown = cv2.inRange(hsv, (0, 20, 30), (40, 255, 255))
    mask_green = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
    mask = cv2.bitwise_or(mask_yellow_brown, mask_green)

    if np.count_nonzero(mask) < IMG_SIZE * IMG_SIZE * 0.05:
        mask = cv2.inRange(hsv, (0, 10, 20), (180, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def extract_color_features(hsv, mask):
    hist_h = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [32], [0, 256])
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)

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
    from skimage.feature import local_binary_pattern
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_pixels = lbp[mask > 0]
    if len(lbp_pixels) == 0:
        return np.zeros(n_points + 2)
    hist, _ = np.histogram(lbp_pixels, bins=n_points + 2, range=(0, n_points + 2), density=True)
    return hist


def extract_gradient_features(gray, mask):
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(5)

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    x, y, w, h = cv2.boundingRect(c)
    extent = area / (w * h) if w * h > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0
    aspect_ratio = float(w) / h if h > 0 else 0

    return np.array([area, perimeter, extent, solidity, aspect_ratio])


def extract_features(image):
    """Master feature extraction - identik dengan train.py."""
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


def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    features = extract_features(image)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    prediction = model.predict(features_pca)
    probabilities = model.predict_proba(features_pca)[0]

    predicted_label = encoder.inverse_transform(prediction)[0]

    # Build confidence scores for all classes
    class_labels = encoder.classes_
    confidence_scores = {}
    for i, label in enumerate(class_labels):
        confidence_scores[label] = float(probabilities[i]) * 100
    
    confidence = float(max(probabilities)) * 100
    
    return {
        'predicted_class': predicted_label,
        'confidence': confidence,
        'confidence_scores': confidence_scores,
        'info': LABEL_INFO.get(predicted_label, {})
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create upload folder if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        
        if result is None:
            return jsonify({'error': 'Gagal memproses gambar'}), 500
        
        return jsonify({
            'success': True,
            'result': result,
            'image_url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, atau JPEG'}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

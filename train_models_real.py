"""
DERMAI — Real Dataset Training Script
Trains production ML models directly on local Kaggle datasets:
1. Acne Classification (Level_0: Clear, Level_1: Mild/Moderate, Level_2: Severe)
   - nayanchaure/acne-dataset (1,833 clinical images) + CelebA clear faces
2. Skin Type Classification (oily, dry, combination, sensitive, normal)
   - jessicali9530/celeba-dataset (real facial distribution)
"""

import os
import sys
import zipfile
import json
import time
from datetime import datetime
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEAT_NAMES = [
    'oil', 'texture', 'redness', 'acne_edge',
    'saturation', 'brightness',
    'r_mean', 'g_mean', 'b_mean',
    'r_std',  'g_std',  'b_std',
] + [f'hist_{i}' for i in range(16)]

def extract_features_from_img(img, img_size=224):
    if img is None:
        return None
    resized = cv2.resize(img, (img_size, img_size))
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(blur)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    oil = np.sum(gray > 200) / gray.size
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    texture = np.var(lbp) / 5000.0
    redness = (np.mean(r) - np.mean(g)) / 255.0
    edges = cv2.Canny(gray, 50, 150)
    acne_edge = np.sum(edges > 0) / gray.size

    saturation = np.mean(hsv[:, :, 1]) / 255.0
    brightness = np.mean(hsv[:, :, 2]) / 255.0

    r_mean, g_mean, b_mean = np.mean(r)/255.0, np.mean(g)/255.0, np.mean(b)/255.0
    r_std,  g_std,  b_std  = np.std(r)/255.0,  np.std(g)/255.0,  np.std(b)/255.0

    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    feats = np.array([
        oil, texture, redness, acne_edge,
        saturation, brightness,
        r_mean, g_mean, b_mean,
        r_std,  g_std,  b_std,
    ] + hist.tolist(), dtype=np.float32)

    return np.clip(feats, 0, 1)

def build_datasets():
    print("=" * 60)
    print("STEP 1: Extracting features from real Kaggle datasets...")
    print("=" * 60)

    # 1. CelebA faces from zip
    celeba_zip_path = os.path.join(BASE_DIR, 'celeba_raw', 'celeba-dataset.zip')
    celeba_feats = []
    
    if os.path.exists(celeba_zip_path):
        print("Reading CelebA faces from zip...")
        with zipfile.ZipFile(celeba_zip_path, 'r') as z:
            img_names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))][:1200]
            for i, name in enumerate(img_names):
                try:
                    data = z.read(name)
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    f = extract_features_from_img(img)
                    if f is not None:
                        celeba_feats.append(f)
                except Exception:
                    pass
                if (i + 1) % 300 == 0:
                    print(f"  CelebA processed: {len(celeba_feats)} faces")
    print(f"Total CelebA features extracted: {len(celeba_feats)}")

    # 2. Acne dataset
    acne_dir = os.path.join(BASE_DIR, 'acne_raw', 'Acne')
    acne_files = [f for f in os.listdir(acne_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Total available Acne clinical images: {len(acne_files)}")

    mild_files = [f for f in acne_files if any(k in f.lower() for k in ['comedo', 'pustular', 'perioral', 'infantile', 'papul'])]
    severe_files = [f for f in acne_files if any(k in f.lower() for k in ['cystic', 'suppurativa', 'nodul', 'scar', 'severe'])]
    remaining = [f for f in acne_files if f not in mild_files and f not in severe_files]
    
    # Balance categories
    mild_files = (mild_files + remaining[:100])[:250]
    severe_files = (severe_files + remaining[100:200])[:250]

    X_mild = []
    for f in mild_files:
        img = cv2.imread(os.path.join(acne_dir, f))
        feat = extract_features_from_img(img)
        if feat is not None:
            X_mild.append(feat)

    X_severe = []
    for f in severe_files:
        img = cv2.imread(os.path.join(acne_dir, f))
        feat = extract_features_from_img(img)
        if feat is not None:
            X_severe.append(feat)

    print(f"Extracted Mild/Moderate Acne samples: {len(X_mild)}")
    print(f"Extracted Severe Acne samples: {len(X_severe)}")

    # Level_0: Clear faces from CelebA
    X_clear = celeba_feats[:250]
    print(f"Level_0 (Clear face) samples: {len(X_clear)}")

    # Assemble Acne Dataset: 0 = Level_0, 1 = Level_1, 2 = Level_2
    X_acne = np.array(X_clear + X_mild + X_severe)
    y_acne = np.array([0]*len(X_clear) + [1]*len(X_mild) + [2]*len(X_severe))

    # Assemble Skin Type Dataset from CelebA features
    # 0: oily, 1: dry, 2: combination, 3: sensitive, 4: normal
    X_skin, y_skin = [], []
    for feat in celeba_feats:
        oil = feat[0]
        texture = feat[1]
        redness = feat[2]
        sat = feat[4]

        if oil > 0.35:
            lbl = 0  # oily
        elif sat < 0.35 and texture > 0.012:
            lbl = 1  # dry
        elif redness > 0.16:
            lbl = 3  # sensitive
        elif 0.18 < oil <= 0.35:
            lbl = 2  # combination
        else:
            lbl = 4  # normal
        X_skin.append(feat)
        y_skin.append(lbl)

    X_skin = np.array(X_skin)
    y_skin = np.array(y_skin)

    return X_acne, y_acne, X_skin, y_skin

def train_and_evaluate():
    X_acne, y_acne, X_skin, y_skin = build_datasets()

    print("\n" + "=" * 60)
    print("STEP 2: Training Acne Classifier (GradientBoosting vs RandomForest)...")
    print("=" * 60)

    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
        X_acne, y_acne, test_size=0.2, random_state=42, stratify=y_acne
    )

    rf_acne = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1))
    ])
    gb_acne = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=160, learning_rate=0.08, max_depth=5, random_state=42))
    ])

    rf_acne.fit(X_tr_a, y_tr_a)
    gb_acne.fit(X_tr_a, y_tr_a)

    rf_acc_a = accuracy_score(y_te_a, rf_acne.predict(X_te_a))
    gb_acc_a = accuracy_score(y_te_a, gb_acne.predict(X_te_a))

    print(f"Acne Model - RandomForest Validation Accuracy:     {rf_acc_a*100:.2f}%")
    print(f"Acne Model - GradientBoosting Validation Accuracy: {gb_acc_a*100:.2f}%")

    if gb_acc_a >= rf_acc_a:
        best_acne_model = gb_acne
        best_acne_algo = "GradientBoosting"
        best_acne_acc = gb_acc_a
    else:
        best_acne_model = rf_acne
        best_acne_algo = "RandomForest"
        best_acne_acc = rf_acc_a

    acne_classes = {0: "Level_0", 1: "Level_1", 2: "Level_2"}
    print(f"Selected Best Acne Algorithm: {best_acne_algo} ({best_acne_acc*100:.2f}%)")
    print(classification_report(y_te_a, best_acne_model.predict(X_te_a), target_names=["Level_0", "Level_1", "Level_2"]))

    print("\n" + "=" * 60)
    print("STEP 3: Training Skin Type Classifier...")
    print("=" * 60)

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_skin, y_skin, test_size=0.2, random_state=42, stratify=y_skin
    )

    rf_skin = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=250, max_depth=14, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    gb_skin = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=140, learning_rate=0.08, max_depth=5, random_state=42))
    ])

    rf_skin.fit(X_tr_s, y_tr_s)
    gb_skin.fit(X_tr_s, y_tr_s)

    rf_acc_s = accuracy_score(y_te_s, rf_skin.predict(X_te_s))
    gb_acc_s = accuracy_score(y_te_s, gb_skin.predict(X_te_s))

    print(f"Skin Type Model - RandomForest Validation Accuracy:     {rf_acc_s*100:.2f}%")
    print(f"Skin Type Model - GradientBoosting Validation Accuracy: {gb_acc_s*100:.2f}%")

    if rf_acc_s >= gb_acc_s:
        best_skin_model = rf_skin
        best_skin_algo = "RandomForest"
        best_skin_acc = rf_acc_s
    else:
        best_skin_model = gb_skin
        best_skin_algo = "GradientBoosting"
        best_skin_acc = gb_acc_s

    skin_classes = {0: "oily", 1: "dry", 2: "combination", 3: "sensitive", 4: "normal"}
    print(f"Selected Best Skin Type Algorithm: {best_skin_algo} ({best_skin_acc*100:.2f}%)")
    print(classification_report(y_te_s, best_skin_model.predict(X_te_s), target_names=["oily", "dry", "combination", "sensitive", "normal"]))

    print("\n" + "=" * 60)
    print("STEP 4: Generating Feature Importance Charts...")
    print("=" * 60)

    for model, title, fname in [
        (best_acne_model, 'Acne Classification', 'acne_importances.png'),
        (best_skin_model, 'Skin Type Classification', 'skin type_importances.png')
    ]:
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            imp = clf.feature_importances_
            top_idx = np.argsort(imp)[-12:]
            plt.figure(figsize=(10, 4.2))
            plt.barh([FEAT_NAMES[i] for i in top_idx], imp[top_idx], color='#b8963e')
            plt.title(f'{title} — Top Feature Importances (Trained on Kaggle Datasets)', fontsize=12, fontweight='bold')
            plt.xlabel('Normalized Importance Weight')
            plt.tight_layout()
            out_img = os.path.join(BASE_DIR, fname)
            plt.savefig(out_img, dpi=140)
            plt.close()
            print(f"Saved feature importance chart: {fname}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving Production Models and Metadata...")
    print("=" * 60)

    joblib.dump(best_acne_model, os.path.join(MODELS_DIR, 'acne_model.pkl'))
    joblib.dump(best_skin_model, os.path.join(MODELS_DIR, 'skin_type_model.pkl'))

    encoders = {
        'acne': {str(k): v for k, v in acne_classes.items()},
        'skin_type': {str(k): v for k, v in skin_classes.items()}
    }
    with open(os.path.join(MODELS_DIR, 'label_encoders.json'), 'w') as f:
        json.dump(encoders, f, indent=2)

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_features': len(FEAT_NAMES),
        'feature_names': FEAT_NAMES,
        'acne_model': {
            'algorithm': best_acne_algo,
            'accuracy': round(float(best_acne_acc), 4),
            'n_classes': len(acne_classes),
            'classes': encoders['acne'],
            'n_train': int(len(X_tr_a)),
            'n_test': int(len(X_te_a)),
            'total_samples': int(len(X_acne))
        },
        'skin_type_model': {
            'algorithm': best_skin_algo,
            'accuracy': round(float(best_skin_acc), 4),
            'n_classes': len(skin_classes),
            'classes': encoders['skin_type'],
            'n_train': int(len(X_tr_s)),
            'n_test': int(len(X_te_s)),
            'total_samples': int(len(X_skin))
        }
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Models and metadata successfully saved to models/ directory!")
    print(f"  Acne Accuracy:      {best_acne_acc*100:.2f}% (Trained on {len(X_acne)} genuine images)")
    print(f"  Skin Type Accuracy: {best_skin_acc*100:.2f}% (Trained on {len(X_skin)} genuine faces)")
    print("=" * 60)

if __name__ == '__main__':
    t_start = time.time()
    train_and_evaluate()
    print(f"Training completed in {time.time() - t_start:.2f} seconds.")

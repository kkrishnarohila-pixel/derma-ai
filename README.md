# DERMAI — Skin Intelligence Platform
### All data is self-trained from Kaggle datasets. Zero hardcoded ingredient knowledge.

---

## Data Flow

```
KAGGLE DATASETS
│
├── nayanchaure/acne-dataset
│   └── [Colab 01] extract_features() → train RandomForest/GradientBoosting
│                                     → acne_model.pkl
│
├── jessicali9530/celeba-dataset
│   └── [Colab 01] extract_features() + rule-based labels
│                                     → skin_type_model.pkl
│
├── kingabzpro/cosmetics-datasets
│   └── [Colab 02] Ingredients column × skin-type columns (0/1)
│                  → ingredient ↔ skin-type good/bad mapping
│                  → category inference from ingredient names
│                                     → ingredient_db.json (partial)
│
└── nadyinky/sephora-products-and-skincare-reviews
    └── [Colab 02] Join products+reviews → avg rating per ingredient × skin-type
                   → enrich ingredient_db.json with review signal
                                         → ingredient_db.json (final)

TRAINED MODELS (place in dermai/models/)
├── acne_model.pkl
├── skin_type_model.pkl
├── label_encoders.json
├── model_metadata.json
└── ingredient_db.json

RUNTIME
├── Face image → skin_analyzer.py → loads acne_model.pkl + skin_type_model.pkl
│                                 → same feature extraction as training
│                                 → skin_type, has_acne, acne_severity
│
├── Product label → ingredient_analyzer.py → Tesseract OCR
│                                           → match against ingredient_db.json
│                                           → matched_ingredients
│
└── Skin + Ingredients → scoring_engine.py → loads ingredient_db.json
                                           → score, verdict, recommendations
```

---

## Setup: Step by Step

### 1. Run Colab Notebooks (Google Colab — free GPU/CPU)

**Notebook 01 — Skin Models**
```
Open: notebooks/01_train_skin_model.ipynb in Google Colab
1. Upload your kaggle.json (from kaggle.com → Settings → API)
2. Run all cells
3. Download skin_models.zip
4. Extract into dermai/models/
   → acne_model.pkl
   → skin_type_model.pkl
   → label_encoders.json
   → model_metadata.json
```

**Notebook 02 — Ingredient Database**
```
Open: notebooks/02_build_ingredient_db.ipynb in Google Colab
1. Upload your kaggle.json
2. Run all cells
3. Download ingredient_db.zip
4. Extract into dermai/models/
   → ingredient_db.json
```

### 2. Install Dependencies

```bash
# System: Tesseract OCR
sudo apt-get install tesseract-ocr   # Ubuntu/Debian
brew install tesseract               # macOS

# Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run

```bash
python app.py
```
Open http://localhost:5000

At startup, the app reports which models are loaded:
```
  Acne model:      ✓ loaded
  Skin-type model: ✓ loaded
  Ingredient DB:   ✓ 2847 ingredients
```

---

## Project Structure

```
dermai/
├── app.py                              Flask server + REST API
├── requirements.txt
├── README.md
│
├── modules/
│   ├── model_loader.py                 Loads .pkl + .json at startup
│   ├── skin_analyzer.py                CV pipeline → trained model inference
│   ├── ingredient_analyzer.py          Tesseract OCR → DB matching
│   ├── scoring_engine.py               Scoring from dataset-derived DB
│   └── models.py                       SQLAlchemy (User, Analysis, Product)
│
├── templates/
│   └── index.html                      Full SPA (Landing → Auth → Hub → Skin → Lab)
│
├── static/uploads/
│
├── models/                             ← Populate after running Colab notebooks
│   ├── acne_model.pkl                  (from notebook 01)
│   ├── skin_type_model.pkl             (from notebook 01)
│   ├── label_encoders.json             (from notebook 01)
│   ├── model_metadata.json             (from notebook 01)
│   └── ingredient_db.json              (from notebook 02)
│
└── notebooks/
    ├── 01_train_skin_model.ipynb       Acne + Skin-type training
    └── 02_build_ingredient_db.ipynb    Ingredient DB from Cosmetics + Sephora
```

---

## API

| Method | Endpoint | Notes |
|--------|----------|-------|
| GET  | `/api/status` | Shows which models are loaded |
| POST | `/api/auth/register` | |
| POST | `/api/auth/login` | |
| GET  | `/api/auth/me` | History |
| POST | `/api/skin/analyse` | `face_image` (base64 or file) |
| POST | `/api/product/analyse` | `product_image` or `ingredients_text` |
| POST | `/api/product/score` | `skin_result` + `ingredient_result` |
| POST | `/api/product/save` | Logged-in only |
| GET  | `/api/ingredients` | Dataset-derived ingredient DB |
| GET  | `/api/skin-profiles` | Dataset-derived profiles |

---

## Kaggle Datasets Used

| Dataset | Purpose | Link |
|---------|---------|------|
| Acne Dataset (nayanchaure) | Acne classification model | https://www.kaggle.com/datasets/nayanchaure/acne-dataset |
| CelebA (jessicali9530) | Skin type feature distribution | https://www.kaggle.com/datasets/jessicali9530/celeba-dataset |
| CelebAMask-HQ (vincenttamml) | Optional: skin segmentation | https://www.kaggle.com/datasets/vincenttamml/celebamaskhq512 |
| Cosmetics (kingabzpro) | Ingredient→skin-type mapping | https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets |
| Sephora Reviews (nadyinky) | Ingredient rating by skin type | https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 / CSS3 / Vanilla JS SPA |
| Backend | Flask + Flask-SQLAlchemy |
| Database | SQLite (dev) / PostgreSQL (prod via DATABASE_URL) |
| CV | OpenCV + MediaPipe Face Mesh |
| Features | Oil (brightness threshold), LBP texture, Canny edges, HSV saturation |
| ML | Scikit-learn RandomForest + GradientBoosting |
| OCR | Tesseract + pytesseract |
| Training | Google Colab (free) |
| Data | Pandas + NumPy |

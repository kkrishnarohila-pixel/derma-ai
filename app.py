"""
DERMAI — Flask Backend
All skin analysis and ingredient scoring uses trained models + dataset-derived data.
No hardcoded ingredient knowledge anywhere in this file.
"""

import os, json, base64, hashlib, secrets, logging
# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

from modules.model_loader       import models_ready, get_ingredient_db, get_skin_profiles, get_metadata
from modules.skin_analyzer      import analyze_skin
from modules.ingredient_analyzer import analyze_product_image, analyze_text_ingredients
from modules.scoring_engine     import full_analysis, score_product, generate_recommendations
from modules.models             import db, User, SkinAnalysis, ProductResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config.update(
    SECRET_KEY              = os.environ.get('SECRET_KEY', secrets.token_hex(32)),
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///dermai.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024,
    UPLOAD_FOLDER           = os.path.join(os.path.dirname(__file__), 'static', 'uploads'),
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
db.init_app(app)

with app.app_context():
    db.create_all()
    status = models_ready()
    logger.info(f'Models status: {status}')


# ── helpers ───────────────────────────────────────────────────────────────────

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def _img_bytes(field):
    if field in request.files:
        f = request.files[field]
        if f and f.filename:
            return f.read()
    data = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
    elif request.form and field in request.form:
        data = {field: request.form[field]}
    elif request.data:
        try:
            data = json.loads(request.data.decode('utf-8'))
        except Exception:
            data = {}
    if data and isinstance(data, dict):
        b64 = data.get(field, '')
        if b64:
            if ',' in b64:
                b64 = b64.split(',')[1]
            return base64.b64decode(b64)
    return None


def _current_user():
    uid = session.get('user_id')
    return User.query.get(uid) if uid else None


# ── SPA ───────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── status ────────────────────────────────────────────────────────────────────

@app.route('/api/status')
def api_status():
    mr = models_ready()
    meta = get_metadata()
    return jsonify({
        'status':        'ok',
        'service':       'DERMAI Skin Intelligence API',
        'version':       '2.0.0',
        'models':        mr,
        'metadata':      meta,
        'trained_at':    meta.get('trained_at', 'not trained yet'),
        'ready':         mr['acne_model'] and mr['skin_type_model'] and mr['ingredient_db'],
        'setup_needed':  not (mr['acne_model'] and mr['skin_type_model'] and mr['ingredient_db']),
        'setup_message': (
            'Run training script or place model files in models/'
            if not (mr['acne_model'] and mr['ingredient_db']) else 'All models loaded ✓'
        ),
    })


# ── auth ──────────────────────────────────────────────────────────────────────

@app.route('/api/auth/register', methods=['POST'])
def register():
    data  = request.get_json(silent=True) or {}
    name  = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    pw    = data.get('password', '')
    if not name or not email or not pw:
        return jsonify({'success': False, 'error': 'Name, email, and password required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'Email already registered'}), 409
    user = User(name=name, email=email, password_hash=_hash(pw))
    db.session.add(user); db.session.commit()
    session['user_id'] = user.id
    return jsonify({'success': True, 'user': user.to_dict()})


@app.route('/api/auth/login', methods=['POST'])
def login():
    data  = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    pw    = data.get('password', '')
    user  = User.query.filter_by(email=email, password_hash=_hash(pw)).first()
    if not user:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    session['user_id'] = user.id
    return jsonify({'success': True, 'user': user.to_dict()})


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})


@app.route('/api/auth/me')
def me():
    user = _current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    analyses = [a.to_dict() for a in
                SkinAnalysis.query.filter_by(user_id=user.id)
                .order_by(SkinAnalysis.created_at.desc()).limit(10).all()]
    products = [p.to_dict() for p in
                ProductResult.query.filter_by(user_id=user.id)
                .order_by(ProductResult.created_at.desc()).limit(20).all()]
    return jsonify({'success': True, 'user': user.to_dict(),
                    'analyses': analyses, 'products': products})


# ── skin analysis ─────────────────────────────────────────────────────────────

@app.route('/api/skin/analyse', methods=['POST'])
def api_skin_analyse():
    img = _img_bytes('face_image')
    if not img:
        return jsonify({'success': False, 'error': 'No face image provided'}), 400

    result = analyze_skin(img)

    if result.get('success'):
        result['recommendations'] = generate_recommendations(
            result['skin_type'],
            result.get('conditions', [result['skin_type']]),
            result.get('has_acne', False)
        )

    user = _current_user()
    if user and result.get('success'):
        row = SkinAnalysis(
            user_id      = user.id,
            skin_type    = result['skin_type'],
            has_acne     = result['has_acne'],
            acne_severity= result['acne_severity'],
            conditions   = json.dumps(result['conditions']),
            scores       = json.dumps(result['scores']),
            confidence   = json.dumps(result['confidence']),
            model_used   = result.get('model_used', False),
        )
        db.session.add(row); db.session.commit()
        result['analysis_id'] = row.id

    return jsonify(result)


# ── ingredient analysis ───────────────────────────────────────────────────────

@app.route('/api/product/analyse', methods=['POST'])
def api_product_analyse():
    img = _img_bytes('product_image')
    if img:
        return jsonify(analyze_product_image(img))
    data = request.get_json(silent=True) or {}
    text = data.get('ingredients_text', '').strip()
    if text:
        return jsonify(analyze_text_ingredients(text))
    return jsonify({'success': False, 'error': 'Provide product_image or ingredients_text'}), 400


# ── scoring ───────────────────────────────────────────────────────────────────

@app.route('/api/product/score', methods=['POST'])
def api_product_score():
    data   = request.get_json(silent=True) or {}
    skin_r = data.get('skin_result')
    ing_r  = data.get('ingredient_result')
    if not skin_r or not ing_r:
        return jsonify({'success': False, 'error': 'skin_result and ingredient_result required'}), 400
    return jsonify(full_analysis(skin_r, ing_r))


# ── save product ──────────────────────────────────────────────────────────────

@app.route('/api/product/save', methods=['POST'])
def api_product_save():
    user = _current_user()
    if not user:
        return jsonify({'success': False, 'error': 'Login required'}), 401
    data = request.get_json(silent=True) or {}
    row  = ProductResult(
        user_id        = user.id,
        product_name   = data.get('product_name', 'Unnamed'),
        ingredients_raw= data.get('ingredients_raw', ''),
        score          = data.get('score', 0),
        verdict        = data.get('verdict', ''),
        breakdown      = json.dumps(data.get('breakdown', {})),
    )
    db.session.add(row); db.session.commit()
    return jsonify({'success': True, 'product_id': row.id})


# ── reference ─────────────────────────────────────────────────────────────────

@app.route('/api/ingredients')
def api_ingredients():
    db_ = get_ingredient_db()
    return jsonify({'success': True, 'total': len(db_), 'data': db_,
                    'source': 'dataset-derived (cosmetics + sephora)'})


@app.route('/api/skin-profiles')
def api_skin_profiles():
    return jsonify({'success': True, 'data': get_skin_profiles(),
                    'source': 'dataset-derived'})


@app.route('/api/samples')
def api_samples():
    """Return genuine sample test data for instant evaluation / demo."""
    samples = []
    sample_files = [
        ('Clear / Healthy Skin', 'clear_face_1.jpg', 'Normal/balanced skin profile sample'),
        ('Clinical Acne Case', 'acne_sample_1.jpg', 'Acne vulgaris / inflammatory breakout sample')
    ]
    samples_dir = os.path.join(os.path.dirname(__file__), 'static', 'samples')
    for label, fname, desc in sample_files:
        fpath = os.path.join(samples_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                b64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('utf-8')
            samples.append({
                'label': label,
                'filename': fname,
                'description': desc,
                'image_b64': b64
            })

    products = [
        {
            'name': 'The Ordinary Niacinamide 10% + Zinc 1%',
            'ingredients': 'water, niacinamide, pentylene glycol, zinc pca, dimethyl isosorbide, tamarindus indica seed gum, xanthan gum, isoceteth-20, ethoxydiglycol, phenoxyethanol, chlorphenesin'
        },
        {
            'name': 'CeraVe Moisturizing Cream',
            'ingredients': 'water, glycerin, cetearyl alcohol, caprylic/capric triglyceride, cetyl alcohol, ceteareth-20, petrolatum, potassium phosphate, ceramide np, ceramide ap, ceramide eop, carbomer, dimethicone, behentrimonium methosulfate, sodium lauroyl lactylate, sodium hyaluronate, cholesterol, phenoxyethanol, disodium edta, dipotassium phosphate, tocopherol, phytosphingosine, xanthan gum, ethylhexylglycerin'
        },
        {
            'name': "Paula's Choice 2% BHA Liquid Exfoliant",
            'ingredients': 'water, methylpropanediol, butylene glycol, salicylic acid, polysorbate 20, camellia oleifera leaf extract, sodium hydroxide, tetrasodium edta'
        }
    ]

    return jsonify({
        'success': True,
        'face_samples': samples,
        'product_samples': products
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large — max 16 MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html')

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    mr = models_ready()
    print('\n' + '='*55)
    print('  DERMAI - Skin Intelligence Platform')
    print('='*55)
    print(f'  Acne model:      {"[OK] loaded" if mr["acne_model"]     else "[X] missing (run notebook 01)"}')
    print(f'  Skin-type model: {"[OK] loaded" if mr["skin_type_model"] else "[X] missing (run notebook 01)"}')
    print(f'  Ingredient DB:   {"[OK] " + str(mr["n_ingredients"]) + " ingredients" if mr["ingredient_db"] else "[X] missing (run notebook 02)"}')
    if not (mr['acne_model'] and mr['ingredient_db']):
        print('\n  -> Run Google Colab notebooks first:')
        print('    notebooks/01_train_skin_model.ipynb')
        print('    notebooks/02_build_ingredient_db.ipynb')
        print('    Then place output files in  models/')
    print('='*55)
    print('  http://localhost:5000')
    print('='*55 + '\n')
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

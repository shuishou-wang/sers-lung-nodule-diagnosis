"""
============================================================================
  app.py v4 — Flask Web Diagnostic Interface (Clinician Edition)

  Design principles:
  - Diagnosis system for CLINICIANS, NOT developers — zero technical jargon in UI
  - Flask Session authentication (configurable credentials, 30min timeout)
  - Full Chinese localization, high-aesthetic medical professional UI
  - DeepSeek LLM AI interpretation with multi-turn follow-up
  - Smart file parsing: auto-detect format, row/column, single/multi-patient
  - Full preprocessing pipeline: airPLS -> SG(11,3) -> ZScore -> PCA(14) -> RF
============================================================================
"""
import sys, os, json, io, warnings, traceback
from datetime import timedelta
warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(CODE_DIR), '模型')
WEB_DIR = os.path.join(os.path.dirname(CODE_DIR), 'web')
sys.path.insert(0, CODE_DIR)

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, session, redirect, url_for
from functools import wraps
from pipeline_utils import (airPLS, predict_single, explain_prediction,
                            compute_literature_peak_analysis)
import pipeline_utils
print(f"  pipeline_utils version: {pipeline_utils.__version__}")
from scipy import signal

# ========================================================================
#  LLM Configuration — 双后端：SiliconFlow (主) + Ollama (备)
# ========================================================================
import openai

# --- 后端选择 ---
# "siliconflow" = 云端快速（免费额度2000万token），"ollama" = 本地免费无限
LLM_BACKEND = "siliconflow"

# --- SiliconFlow (OpenAI 兼容) ---
SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY', '')
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 7B参数，中文医学远优于本地3B

# --- Ollama (本地免费备选) ---
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"

# --- 初始化 ---
LLM_ENABLED = True
if LLM_BACKEND == "siliconflow":
    openai.api_key = SILICONFLOW_API_KEY
    openai.api_base = SILICONFLOW_BASE_URL
    ACTIVE_LLM_MODEL = SILICONFLOW_MODEL
    print(f"  LLM: SiliconFlow cloud (model={SILICONFLOW_MODEL})")
else:
    openai.api_key = "ollama"
    openai.api_base = OLLAMA_BASE_URL
    ACTIVE_LLM_MODEL = OLLAMA_MODEL
    print(f"  LLM: Ollama local (model={OLLAMA_MODEL})")

# ========================================================================
#  System Prompt for LLM Diagnosis Interpretation
# ========================================================================
LLM_SYSTEM_PROMPT = """你是SERS多模态融合肺结节诊断系统的AI临床顾问。你的角色是帮助临床医生和研究人员理解诊断结果。

## 系统背景
- 技术路线：表面增强拉曼光谱(SERS) + 临床肿瘤标志物(CEA/SCC/NSE) → 随机森林中融合
- 训练数据：49例（恶性30例，良性19例）
- 验证方式：留一交叉验证(LOOCV)
- 性能指标：准确率87.8%，AUC 0.891，灵敏度86.7%，特异度89.5%

## 文献验证的拉曼生物标志物
以下6个峰均来自肺癌Meta分析文献，且在我们的数据中均通过Welch's t-test(p<0.05)：

1. **643 cm⁻¹ — 酪氨酸C-C扭转**
   机制：酪氨酸激酶受体过度激活 → 磷酸化↑ → 细胞增殖失控

2. **822 cm⁻¹ — 酪氨酸环呼吸(面外)**
   机制：酪氨酸代谢重编程 → 肿瘤微环境重塑

3. **1004 cm⁻¹ — 苯丙氨酸环呼吸 ★**
   机制：Warburg效应 → 氨基酸代谢重编程 → 苯丙氨酸转运/利用↑

4. **1126 cm⁻¹ — 蛋白C-N伸缩 / 脂质C-C**
   机制：细胞膜合成↑ + 蛋白质合成代谢↑ → 增殖和侵袭

5. **1655 cm⁻¹ — 酰胺I(α-螺旋) ★**
   机制：蛋白质二级结构改变 → 癌蛋白过表达

6. **1675 cm⁻¹ — 酰胺I(β-折叠) ★**
   机制：β-折叠蛋白聚集 → 癌相关成纤维细胞活化

## 临床标志物
- CEA(癌胚抗原)：肺腺癌等多种癌症中升高
- SCC(鳞状细胞癌抗原)：肺鳞癌中升高
- NSE(神经元特异性烯醇化酶)：小细胞肺癌中升高

## 回复要求
1. 用通俗中文解释预测结果，引用具体拉曼峰的生物学意义
2. 峰显示"↑恶性"时说明可能提示什么生物过程；显示"↓良性"时说明可能意味着什么
3. 如果多个峰都指向同一个生物学通路（如蛋白质代谢异常），要主动归纳
4. 承认不确定性——这是筛查辅助工具，不是确诊金标准
5. 初始解读2-4段，追问答复简洁直接
6. 所有回复使用中文"""


def build_chat_messages(diagnosis_context):
    """Build the messages array for the initial diagnosis interpretation."""
    prob = diagnosis_context.get('probability_malignant', 0) * 100
    pred = diagnosis_context.get('prediction', 'Unknown')
    clinical = diagnosis_context.get('clinical_drivers', [])
    spectral = diagnosis_context.get('spectral_drivers', [])

    # Format clinical drivers
    clin_lines = []
    for c in clinical:
        clin_lines.append(
            f"- {c['name']}: Z-score={c['value_z']}, "
            f"重要性={c['importance_pct']}%, 方向={c['direction']}"
        )

    # Format spectral drivers
    spec_lines = []
    for s in spectral:
        spec_lines.append(
            f"- {s['name']} ({s['shift_cm']:.0f} cm-1): "
            f"样本Z-score={s['value_z']}, 重要性={s['importance_pct']}%, "
            f"方向={s['direction']}, p={s.get('p_value', 0):.2e}\n"
            f"  生物机制: {s.get('bio_mechanism', '')}"
        )

    user_message = f"""请解读以下诊断结果：

## 预测结果
- 恶性概率: {prob:.1f}%
- 良性概率: {100-prob:.1f}%
- 综合判断: {pred}

## 临床标志物贡献
{chr(10).join(clin_lines) if clin_lines else '无临床数据'}

## 拉曼特征峰贡献（文献验证）
{chr(10).join(spec_lines) if spec_lines else '无显著峰'}

请给出你的解读。"""

    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ========================================================================
#  Load Pipeline & Assets
# ========================================================================
print("=" * 60)
print("  Loading SERS 多模态融合肺结节诊断系统 v4...")

PIPELINE = joblib.load(os.path.join(MODEL_DIR, 'pipeline.pkl'))

# Load Raman shifts from training data
CACHE = np.load(os.path.join(MODEL_DIR, 'train_data_cache.npz'), allow_pickle=True)
ram_shifts = CACHE['raman_shifts']
PIPELINE['raman_shifts'] = np.array([float(str(s).replace(',', '.')) if isinstance(s, str) else float(s)
                                      for s in ram_shifts])

# Compute literature-validated peak analysis (literature-first approach)
# Only peaks that are BOTH (a) known lung cancer biomarkers from meta-analyses
# AND (b) statistically significant (Welch's t-test p<0.05) are retained.
_sers = CACHE['sers_scaled']
_y = CACHE['y']
_lit = compute_literature_peak_analysis(_sers, _y, PIPELINE['raman_shifts'], p_thresh=0.05)
PIPELINE['literature_peak_analysis'] = _lit
print(f"  Literature peaks: {_lit['n_significant']}/{len(pipeline_utils.LITERATURE_LUNG_CANCER_PEAKS)} pass t-test (p<0.05)")
for pk in _lit['peaks']:
    print(f"    {pk['label']}  p={pk['p_value']:.2e}  importance={pk['importance_pct']}%")

DEMO_CASES = []
demo_path = os.path.join(MODEL_DIR, 'demo_cases.json')
if os.path.exists(demo_path):
    with open(demo_path, 'r', encoding='utf-8') as f:
        DEMO_CASES = json.load(f)

# Default clinical values (training median of group medians)
DEPS = {}
for feat in ['CEA', 'SCC', 'NSE']:
    imputer = PIPELINE['clinical_imputer']
    meds = imputer.medians_.get(feat, {'malignant': 0, 'benign': 0})
    DEPS[feat] = round(np.nanmedian([meds['malignant'], meds['benign']]), 2)

print(f"  Pipeline: PCA={PIPELINE['pca_n']}D, RF({PIPELINE['rf_n_estimators']},{PIPELINE['rf_max_depth']})")
print(f"  Raman shifts: {len(PIPELINE['raman_shifts'])} loaded")
print(f"  Defaults: CEA={DEPS['CEA']}, SCC={DEPS['SCC']}, NSE={DEPS['NSE']}")
print(f"  Demo cases: {len(DEMO_CASES)}")

# ========================================================================
#  Flask App
# ========================================================================
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'change-me-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
os.makedirs(os.path.join(WEB_DIR, 'templates'), exist_ok=True)


# ========================================================================
#  Authentication
# ========================================================================
USERNAME = 'admin'
PASSWORD = os.environ.get('APP_PASSWORD', 'admin123')


def login_required(f):
    """Decorator: redirect to /login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': '请先登录', 'redirect': '/login'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def preprocess_for_display(raw_1d):
    """airPLS -> SG(11,3) only (before ZScore), for spectrum display."""
    raw = np.asarray(raw_1d, dtype=float)
    corr, _ = airPLS(raw)
    w = PIPELINE['sg_window']; p2 = PIPELINE['sg_polyorder']
    if w % 2 == 0: w -= 1
    return signal.savgol_filter(corr, w, p2)


def full_preprocess(raw_1d):
    """Full preprocessing: airPLS -> SG -> ZScore -> PCA. Returns (pca_coords, scaled)."""
    raw = np.asarray(raw_1d, dtype=float)
    corr, _ = airPLS(raw)
    w = PIPELINE['sg_window']; p2 = PIPELINE['sg_polyorder']
    if w % 2 == 0: w -= 1
    smoothed = signal.savgol_filter(corr, w, p2)
    scaled = PIPELINE['sers_scaler'].transform(smoothed.reshape(1, -1))
    pca_coords = PIPELINE['sers_pca'].transform(scaled)
    return pca_coords[0], scaled[0]


def parse_spectrum_file(file_content, filename):
    """Smart spectrum file parser — auto-detects format, orientation, multi-patient.

    Returns dict:
      success: bool
      spectra: [[float,...], ...]   — list of spectra
      n_spectra: int
      n_values: int
      message: str
      format: str
    """
    fn = filename.lower()

    # ---- Excel ----
    if fn.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(io.BytesIO(file_content))
        except Exception as e:
            return {'success': False, 'error': f'Excel read failed: {e}'}
        return _parse_dataframe(df, filename)

    # ---- CSV ----
    if fn.endswith('.csv'):
        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding=enc)
                break
            except:
                continue
        else:
            return {'success': False, 'error': 'CSV read failed (tried utf-8, gbk, latin-1)'}
        return _parse_dataframe(df, filename)

    # ---- Plain text (.txt) ----
    try:
        text = file_content.decode('utf-8')
    except:
        text = file_content.decode('gbk', errors='ignore')

    import re
    numbers = []
    for line in text.strip().split('\n'):
        for token in re.split(r'[,\s\t;]+', line.strip()):
            token = token.strip()
            if not token: continue
            try:
                numbers.append(float(token))
            except ValueError:
                pass

    if len(numbers) < 10:
        return {'success': False, 'error': f'Found only {len(numbers)} numeric values'}

    if 500 <= len(numbers) <= 2000:
        return {'success': True, 'spectra': [numbers], 'n_spectra': 1,
                'n_values': len(numbers), 'format': 'text_single',
                'message': f'Single spectrum ({len(numbers)} values)'}

    return {'success': False,
            'error': f'Found {len(numbers)} values — expected ~1000 per spectrum'}


def _parse_dataframe(df, filename):
    """Parse DataFrame into spectra. Handles: labeled columns, shift column, row/col orientation."""
    n_rows, n_cols = df.shape

    # ---- Detect Raman shift column ----
    raman_keywords = ['raman', 'shift', 'cm', 'wavenumber', 'intensity', '拉曼', '位移', '强度', '波数']
    has_shift_col = False
    shift_col_idx = 0

    # Check header keywords
    headers = [str(c).lower() for c in df.columns]
    for i, h in enumerate(headers):
        if any(kw in h for kw in raman_keywords):
            if i == 0:
                has_shift_col = True; shift_col_idx = i
            break

    # Check first column for monotonically increasing values in Raman range
    if not has_shift_col:
        try:
            first_col = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            if first_col.notna().sum() > max(10, n_rows * 0.8):
                vals = first_col.dropna().values
                if len(vals) > 10 and np.all(np.diff(vals[:min(50, len(vals))]) > 0):
                    if 300 < vals[0] < 2000 and 300 < vals[-1] < 2000:
                        has_shift_col = True
        except:
            pass

    # ---- Extract spectra ----
    spectra = []
    message = ''

    if has_shift_col or any(any(kw in str(c).lower() for kw in raman_keywords) for c in df.columns):
        # Column-oriented: each data column (after shift) = one spectrum
        start = 1 if has_shift_col else 0
        cols = [c for c in df.columns[start:] if not any(kw in str(c).lower() for kw in raman_keywords)]
        if not cols:
            cols = df.columns[start:]
        for col in cols:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(vals) >= 100:
                spectra.append(vals[:1000].tolist())
        message = f'{len(spectra)} spectrum column(s)'
        if has_shift_col:
            message += ', with Raman shift column'

    else:
        # All numeric — detect orientation
        if n_cols > n_rows and n_cols > 100:
            # Rows = spectra
            for i in range(min(n_rows, 500)):
                row = pd.to_numeric(df.iloc[i], errors='coerce').dropna().values
                if len(row) >= 100:
                    spectra.append(row[:1000].tolist())
            message = f'{len(spectra)} spectra (row vectors, {n_cols} values each)'
        elif n_rows > n_cols and n_rows > 100:
            # Columns = spectra
            for col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna().values
                if len(vals) >= 100:
                    spectra.append(vals[:1000].tolist())
            message = f'{len(spectra)} spectra (column vectors, {n_rows} values each)'
        else:
            # Ambiguous — try both, pick shape closer to 1000
            if abs(n_cols - 1000) < abs(n_rows - 1000) and n_cols > 10:
                for i in range(min(n_rows, 500)):
                    row = pd.to_numeric(df.iloc[i], errors='coerce').dropna().values
                    if len(row) >= 100:
                        spectra.append(row[:1000].tolist())
                message = f'{len(spectra)} spectra (rows)'
            elif n_rows > 10:
                for col in df.columns:
                    vals = pd.to_numeric(df[col], errors='coerce').dropna().values
                    if len(vals) >= 100:
                        spectra.append(vals[:1000].tolist())
                message = f'{len(spectra)} spectra (columns)'

    if not spectra:
        return {'success': False, 'error': 'No spectra detected. Check file format.'}

    # Normalize lengths
    lens = [len(s) for s in spectra]
    min_len = min(lens)
    if min_len < 100:
        return {'success': False, 'error': f'Spectra too short ({min_len} values)'}
    if max(lens) != min_len:
        spectra = [s[:min_len] for s in spectra]

    # If fewer than 1000, pad with last value to avoid pipeline mismatch
    if min_len < 1000:
        for i in range(len(spectra)):
            pad_val = spectra[i][-1] if spectra[i] else 0
            spectra[i].extend([pad_val] * (1000 - len(spectra[i])))
        message += f' (padded to 1000 from {min_len})'

    return {
        'success': True,
        'spectra': spectra,
        'n_spectra': len(spectra),
        'n_values': len(spectra[0]),
        'message': message,
        'format': 'dataframe',
    }


# ========================================================================
#  Auth Routes
# ========================================================================

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Login page — GET shows form, POST validates credentials."""
    if request.method == 'POST':
        data = request.get_json(force=True) if request.is_json else request.form
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            session.permanent = True
            return jsonify({'success': True, 'redirect': '/'})
        else:
            return jsonify({'success': False, 'error': '账号或密码错误'}), 401

    # GET: serve login page
    login_html_path = os.path.join(WEB_DIR, 'templates', 'login.html')
    if os.path.exists(login_html_path):
        with open(login_html_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Fallback: minimal inline login page
    return _inline_login_html()


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/api/session')
def check_session():
    """Check if user is logged in (for client-side auth checks)."""
    return jsonify({
        'logged_in': session.get('logged_in', False),
        'username': session.get('username', ''),
    })


def _inline_login_html():
    """Minimal inline login page (fallback until login.html is created)."""
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERS 多模态融合肺结节诊断系统 — 登录</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{display:flex;align-items:center;justify-content:center;min-height:100vh;
  background:#f8fafc;font-family:"PingFang SC","Microsoft YaHei",sans-serif}
.login-card{background:#fff;padding:48px 40px;border-radius:12px;
  box-shadow:0 1px 2px rgba(0,0,0,.04),0 2px 8px rgba(0,0,0,.06);
  width:400px;max-width:90vw}
.login-card h1{font-size:22px;color:#0f172a;text-align:center;margin-bottom:8px}
.login-card h2{font-size:15px;color:#64748b;font-weight:400;text-align:center;margin-bottom:32px}
.form-group{margin-bottom:20px}
.form-group label{display:block;font-size:13px;color:#334155;margin-bottom:6px}
.form-group input{width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;
  font-size:14px;color:#0f172a;outline:none;transition:border-color .2s}
.form-group input:focus{border-color:#1e40af;box-shadow:0 0 0 3px rgba(30,64,175,.1)}
.btn-login{width:100%;padding:12px;background:#1e40af;color:#fff;border:none;
  border-radius:8px;font-size:15px;cursor:pointer;font-weight:500;transition:background .2s}
.btn-login:hover{background:#1e3a8a}
.error-msg{color:#d97706;font-size:13px;text-align:center;margin-top:12px;display:none}
</style>
</head>
<body>
<div class="login-card">
<h1>SERS 多模态融合肺结节诊断系统</h1>
<h2>登录</h2>
<div class="form-group">
<label>账号</label>
<input type="text" id="username" placeholder="请输入账号" autocomplete="username">
</div>
<div class="form-group">
<label>密码</label>
<input type="password" id="password" placeholder="请输入密码" autocomplete="current-password">
</div>
<button class="btn-login" onclick="doLogin()">登 录</button>
<p class="error-msg" id="errorMsg"></p>
</div>
<script>
document.getElementById('password').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
async function doLogin(){
  const u=document.getElementById('username').value.trim();
  const p=document.getElementById('password').value.trim();
  const err=document.getElementById('errorMsg');
  if(!u||!p){err.textContent='请输入账号和密码';err.style.display='block';return}
  try{
    const r=await fetch('/login',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username:u,password:p})});
    const d=await r.json();
    if(d.success){window.location.href=d.redirect}
    else{err.textContent=d.error;err.style.display='block'}
  }catch(e){err.textContent='网络异常，请重试';err.style.display='block'}
}
</script>
</body>
</html>'''


# ========================================================================
#  API Endpoints
# ========================================================================

@app.route('/')
@login_required
def index():
    html_path = os.path.join(WEB_DIR, 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    demo_json = json.dumps(DEMO_CASES, ensure_ascii=False)
    defaults_json = json.dumps(DEPS, ensure_ascii=False)
    inject = (
        f'<script>window.DEMO_CASES = {demo_json};'
        f'window.DEFAULTS = {defaults_json};</script>\n</body>'
    )
    return html.replace('</body>', inject)


@app.route('/api/demo_cases')
@login_required
def get_demo_cases():
    return jsonify(DEMO_CASES)


@app.route('/api/defaults')
@login_required
def get_defaults():
    return jsonify(DEPS)


@app.route('/api/parse_spectrum', methods=['POST'])
@login_required
def parse_spectrum():
    """Smart format detection for uploaded spectrum file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    file = request.files['file']
    result = parse_spectrum_file(file.read(), file.filename)
    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    """Single sample prediction: spectrum + optional clinical markers.
    Returns: probability, preprocessed spectrum, diagnostic drivers.
    """
    try:
        data = request.get_json(force=True)

        # Spectrum (required, must be the correct length)
        spectrum_raw = data.get('spectrum', [])
        if len(spectrum_raw) < 100:
            return jsonify({'error': f'Spectrum too short ({len(spectrum_raw)} values)'}), 400

        spectrum_raw = np.array(spectrum_raw, dtype=float)

        # Handle mismatched lengths
        if len(spectrum_raw) != 1000:
            if len(spectrum_raw) > 1000:
                spectrum_raw = spectrum_raw[:1000]
            else:
                pad_val = spectrum_raw[-1]
                spectrum_raw = np.pad(spectrum_raw, (0, 1000 - len(spectrum_raw)),
                                       constant_values=pad_val)

        # Clinical features (optional)
        clinical_provided = all(k in data for k in ['cea', 'scc', 'nse'])
        cea = float(data.get('cea', DEPS['CEA']))
        scc = float(data.get('scc', DEPS['SCC']))
        nse = float(data.get('nse', DEPS['NSE']))

        # ---- Full preprocessing + prediction ----
        pca_coords, scaled_spectrum = full_preprocess(spectrum_raw)

        clin = {'CEA': np.array([cea]), 'SCC': np.array([scc]), 'NSE': np.array([nse])}
        clin_imputed = PIPELINE['clinical_imputer'].transform(clin)
        clin_scaled = PIPELINE['clinical_scaler'].transform(clin_imputed)[0]

        result = predict_single(PIPELINE, spectrum_raw, clin)

        # ---- Display spectrum (airPLS+SG only, no ZScore) ----
        display_spectrum = preprocess_for_display(spectrum_raw)

        # ---- Diagnostic drivers (per-sample, literature-validated peaks) ----
        drivers = explain_prediction(PIPELINE, pca_coords, clin_scaled,
                                     scaled_spectrum, n_top_peaks=6)

        return jsonify({
            'probability_malignant': result['probability_malignant'],
            'probability_benign': result['probability_benign'],
            'prediction': result['prediction'],
            'display_spectrum': display_spectrum.tolist(),
            'raman_shifts': PIPELINE['raman_shifts'].tolist(),
            'cea': cea, 'scc': scc, 'nse': nse,
            'clinical_provided': clinical_provided,
            'clinical_note': '' if clinical_provided else
                f'Clinical defaults used (CEA={DEPS["CEA"]}, SCC={DEPS["SCC"]}, NSE={DEPS["NSE"]})',
            'drivers': drivers,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
@login_required
def predict_batch():
    """Batch prediction: used after user picks a single patient from multi-patient file."""
    try:
        data = request.get_json(force=True)
        spectrum_raw = data.get('spectrum', [])
        if len(spectrum_raw) < 100:
            return jsonify({'error': 'Spectrum too short'}), 400

        # Reuse single predict logic
        data['spectrum'] = spectrum_raw
        return predict()

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """LLM chat endpoint — initial interpretation or follow-up questions.

    Expects JSON:
      - diagnosis_context: dict with probability_malignant, prediction,
                           clinical_drivers, spectral_drivers
      - messages: list of {role, content} for conversation history
      - new_message: str (the user's latest question)
    """
    try:
        if not LLM_ENABLED:
            return jsonify({'error': 'LLM disabled — set DEEPSEEK_API_KEY env var'}), 503

        data = request.get_json(force=True)
        diagnosis_context = data.get('diagnosis_context', {})
        history = data.get('messages', [])
        new_message = data.get('new_message', '')

        # Build full messages array
        if not history:
            # First call: generate initial interpretation
            messages = build_chat_messages(diagnosis_context)
        else:
            # Follow-up: system prompt + history + new message
            messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append({"role": "user", "content": new_message})

        response = openai.ChatCompletion.create(
            model=ACTIVE_LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content

        return jsonify({
            'reply': reply,
            'role': 'assistant',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  SERS 多模态融合肺结节诊断系统 v4")
    print(f"  Open: http://localhost:5000")
    print(f"  Login: admin / (see APP_PASSWORD env var)")
    print(f"  Session: 30min timeout")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)

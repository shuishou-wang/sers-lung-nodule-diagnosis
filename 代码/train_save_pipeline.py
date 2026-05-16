"""
============================================================================
  train_save_pipeline.py
  Train full Mid-Fusion Pipeline on all 49 samples and save for inference.

  Pipeline (opt_C5 / opt_new4 paper-grade):
    SERS:     airPLS -> SG(11,3) -> ZScore -> PCA(14)
    Clinical: CEA+SCC+NSE -> group-median impute -> ZScore
    Fusion:   hstack(14+3=17d) -> SMOTE(k=3) -> RF(50,3)

  Output: ../模型/
    pipeline.pkl           # Full inference pipeline
    pipeline_metadata.json # Hyperparameters and training info
    train_data_cache.npz   # Training data cache (for visualizations)
============================================================================
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')

# Ensure pipeline_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from pipeline_utils import airPLS, SGTransformer, ClinicalImputer

# ====== Paths ======
DATA_PATH = os.environ.get('DATA_PATH', 'data/肿瘤标志物+拉曼数据+年龄性别.xlsx')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '模型')
os.makedirs(MODEL_DIR, exist_ok=True)

# ====== Hyperparameters (opt_C5 / opt_new4 optimal) ======
SG_W, SG_P = 11, 3
PCA_N = 14
SMOTE_K = 3
RF_NE, RF_MD = 50, 3
RANDOM_STATE = 42

# ========================================================================
#  Load Data
# ========================================================================
print("=" * 70)
print("  Loading data & training full pipeline...")
print("=" * 70)

df = pd.read_excel(DATA_PATH)
df.columns = (['Disease', 'Gender', 'Age', 'CEA', 'SCC', 'CYFRA21_1', 'NSE', 'ProGRP']
              + list(df.columns[8:]))
y = df['Disease'].values.astype(int)
sers_raw = df.iloc[:, 8:].apply(pd.to_numeric, errors='coerce').values
raman_shifts = df.columns[8:].tolist()

cea_raw = df['CEA'].values.astype(float)
scc_raw = df['SCC'].values.astype(float)
nse_raw = df['NSE'].values.astype(float)

N = len(y); M = y.sum(); B = N - M
print(f"  Dataset: N={N} (Malignant={M}, Benign={B})")
print(f"  SERS: {sers_raw.shape[1]} Raman shifts, Clinical: CEA+SCC+NSE")

# ========================================================================
#  Train Pipeline (all 49 samples, no LOOCV split)
# ========================================================================

# [1] airPLS baseline correction
print(f"\n  [1/5] airPLS baseline correction ({N} spectra)...")
t0 = time.time()
sers_airpls = np.zeros_like(sers_raw)
for i in range(N):
    sers_airpls[i], _ = airPLS(sers_raw[i])
print(f"        Done ({time.time()-t0:.1f}s)")

# [2] SG smoothing
print(f"  [2/5] SG smoothing (w={SG_W}, p={SG_P})...")
sg = SGTransformer(SG_W, SG_P)
sers_sg = sg.fit_transform(sers_airpls)

# [3] SERS: ZScore + PCA (fit on all data)
print(f"  [3/5] ZScore + PCA(n={PCA_N}) on SERS...")
scl_s = StandardScaler()
sers_scaled = scl_s.fit_transform(sers_sg)

pca = PCA(n_components=PCA_N, random_state=RANDOM_STATE)
sers_pca = pca.fit_transform(sers_scaled)
print(f"        PCA cumvar: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# [4] Clinical: impute + ZScore
print(f"  [4/5] Clinical preprocessing (impute + ZScore)...")
clin_dict = {'CEA': cea_raw, 'SCC': scc_raw, 'NSE': nse_raw}
imputer = ClinicalImputer()
clin_filled = imputer.fit_transform(clin_dict, y)

scl_c = StandardScaler()
clin_scaled = scl_c.fit_transform(clin_filled)

# [5] Fusion + SMOTE + RF
print(f"  [5/5] Mid-Level Fusion: hstack({PCA_N}+3={PCA_N+3}d) -> SMOTE(k={SMOTE_K}) -> RF({RF_NE},{RF_MD})...")
X_fusion = np.hstack([sers_pca, clin_scaled])

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_res, y_res = smote.fit_resample(X_fusion, y)
print(f"        After SMOTE: {X_res.shape[0]} samples")

rf = RandomForestClassifier(
    n_estimators=RF_NE, max_depth=RF_MD,
    class_weight='balanced', random_state=RANDOM_STATE
)
rf.fit(X_res, y_res)

# Quick sanity check
y_pred = rf.predict(X_fusion)
y_prob = rf.predict_proba(X_fusion)[:, 1]
train_acc = accuracy_score(y, y_pred)
train_auc = roc_auc_score(y, y_prob)
print(f"        Training Acc: {train_acc*100:.1f}%  AUC: {train_auc:.4f}")

# ========================================================================
#  Save Pipeline
# ========================================================================
pipeline = {
    'sers_scaler': scl_s,
    'sers_pca': pca,
    'sg_window': SG_W,
    'sg_polyorder': SG_P,
    'clinical_imputer': imputer,
    'clinical_scaler': scl_c,
    'clinical_features': ['CEA', 'SCC', 'NSE'],
    'rf_classifier': rf,
    'airpls_lambda': 1e6,
    'pca_n': PCA_N,
    'smote_k': SMOTE_K,
    'rf_n_estimators': RF_NE,
    'rf_max_depth': RF_MD,
    'random_state': RANDOM_STATE,
}

model_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
joblib.dump(pipeline, model_path, compress=3)
print(f"\n  Pipeline saved: {model_path}")
print(f"  File size: {os.path.getsize(model_path)/1024:.1f} KB")

# ========================================================================
#  Save Metadata
# ========================================================================
metadata = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pipeline': 'airPLS->SG(11,3)->ZScore->PCA(14) || CEA+SCC+NSE->impute->ZScore || hstack->SMOTE(3)->RF(50,3)',
    'hyperparameters': {
        'sg_window': SG_W, 'sg_polyorder': SG_P,
        'pca_n': PCA_N,
        'pca_cumvar': float(pca.explained_variance_ratio_.sum()),
        'smote_k': SMOTE_K,
        'rf_n_estimators': RF_NE, 'rf_max_depth': RF_MD,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
    },
    'dataset': {'n_samples': N, 'malignant': int(M), 'benign': int(B)},
    'data_path': DATA_PATH,
}
with open(os.path.join(MODEL_DIR, 'pipeline_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ========================================================================
#  Save Training Data Cache (for web app visualizations)
# ========================================================================
cache = {
    'y': y,
    'sers_pca': sers_pca,
    'clin_scaled': clin_scaled,
    'sers_scaled': sers_scaled,
    'raman_shifts': np.array(raman_shifts, dtype=object),
    'mean_spectrum_malignant': sers_scaled[y == 1].mean(axis=0),
    'mean_spectrum_benign': sers_scaled[y == 0].mean(axis=0),
}
np.savez_compressed(os.path.join(MODEL_DIR, 'train_data_cache.npz'), **cache)
print(f"  Data cache saved: {os.path.join(MODEL_DIR, 'train_data_cache.npz')}")

# ========================================================================
#  Summary
# ========================================================================
print(f"\n{'=' * 70}")
print(f"  TRAINING COMPLETE")
print(f"{'=' * 70}")
print(f"  Training-set performance (reference only, NOT LOOCV):")
print(f"    Accuracy: {train_acc*100:.1f}%")
print(f"    AUC:      {train_auc:.4f}")
print(f"")
print(f"  Paper-reported LOOCV: Acc=87.8%  AUC=0.891")
print(f"")
print(f"  Ready for: python app.py  (Flask web diagnostic interface)")
print(f"{'=' * 70}")

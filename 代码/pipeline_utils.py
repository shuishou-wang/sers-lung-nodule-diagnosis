"""
============================================================================
  pipeline_utils.py  —  Shared module for Mid-Fusion pipeline components
  Used by: train_save_pipeline.py, Flask app (app.py), inference scripts

  Classes must be importable for joblib deserialization.
============================================================================
"""
__version__ = "2.0-ttest-20260516"
import numpy as np
from scipy import signal
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin


def airPLS(x, lambda_=1e6, max_iter=15, tol=1e-3):
    """Adaptive iteratively reweighted penalized least squares baseline correction.
    Zhang et al., Analyst, 2010.

    Parameters
    ----------
    x : ndarray, 1D spectrum
    lambda_ : float, smoothing parameter (larger = smoother)
    max_iter : int, max iterations
    tol : float, convergence tolerance

    Returns
    -------
    corrected : ndarray, baseline-corrected spectrum (x - baseline)
    baseline : ndarray, fitted baseline
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csc')
    DTD = D @ D.T
    w = np.ones(L)
    for it in range(max_iter):
        W = diags(w, 0, format='csc')
        z = spsolve(W + lambda_ * DTD, w * x)
        d = x - z
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break
        ss = np.sum(np.abs(d_neg))
        if ss < tol:
            break
        w_new = np.zeros(L)
        w_new[d >= 0] = 0
        w_new[d < 0] = np.exp(it * d[d < 0] / ss)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return x - z, z


class SGTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing as sklearn-compatible transformer."""

    def __init__(self, w=9, p=2):
        self.w = w
        self.p = p

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        wl = min(self.w, X.shape[1])
        if wl % 2 == 0:
            wl -= 1
        if wl <= self.p:
            return X
        return np.array([signal.savgol_filter(x, wl, self.p) for x in X])


class ClinicalImputer:
    """Clinical feature imputer storing group medians for inference.

    During training: fits group-specific (malignant/benign) medians.
    During inference (single sample, unknown label): uses overall median fallback.
    """

    def __init__(self):
        self.medians_ = {}
        self.feature_names_ = []

    def fit(self, X_dict, y):
        """Fit group-specific medians from training data.

        Parameters
        ----------
        X_dict : dict of {str: array}
            Clinical features keyed by name.
        y : ndarray of int
            Labels (0=benign, 1=malignant).
        """
        self.feature_names_ = list(X_dict.keys())
        for name, vals in X_dict.items():
            vals = np.asarray(vals, dtype=float)
            mal_med = (np.nanmedian(vals[y == 1])
                       if (y == 1).sum() > 0 else np.nanmedian(vals))
            ben_med = (np.nanmedian(vals[y == 0])
                       if (y == 0).sum() > 0 else np.nanmedian(vals))
            self.medians_[name] = {'malignant': float(mal_med),
                                    'benign': float(ben_med)}
        return self

    def transform(self, X_dict):
        """Fill NaN using overall median (label unknown at inference time).

        Parameters
        ----------
        X_dict : dict of {str: array}
            Clinical features keyed by name with shape (n_samples,).
        """
        result = []
        for name in self.feature_names_:
            vals = np.atleast_1d(np.asarray(X_dict[name], dtype=float))
            med = self.medians_[name]
            fallback = np.nanmedian([med['malignant'], med['benign']])
            filled = np.where(np.isnan(vals), fallback, vals)
            result.append(filled)
        return np.column_stack(result)

    def fit_transform(self, X_dict, y):
        """Fit and transform in one call."""
        self.fit(X_dict, y)
        return self.transform(X_dict)


# ========================================================================
#  Raman peak assignment database (biological samples, 400-1800 cm⁻¹)
# ========================================================================
RAMAN_PEAK_DB = [
    # --- Low wavenumber / Skeletal / Disulfide region ---
    (497,  "S-S disulfide (protein conformation)"),
    (540,  "C-S stretch / S-S gauche"),
    (573,  "Tryptophan / C-S"),
    # --- Amino acid markers ---
    (621,  "Phenylalanine (ring)"),
    (643,  "Tyrosine (C-C ring)"),
    (720,  "DNA Adenine"),
    (760,  "Tryptophan (ring)"),
    (810,  "DNA/RNA O-P-O backbone (phosphodiester)"),
    (830,  "Tyrosine (ring breathing)"),
    (853,  "Tyrosine (ring breathing)"),
    (889,  "Deoxyribose C-C / Collagen"),
    (937,  "Proline/Valine (C-C)"),
    (1004, "Phenylalanine (ring breathing)"),
    (1031, "Phenylalanine (C-H)"),
    # --- Amide III / Protein / DNA region ---
    (1087, "Amide III / PO2- symmetric (DNA)"),
    (1126, "Protein C-N / Lipid C-C"),
    (1134, "Protein C-N / Nucleic acid backbone"),
    (1174, "Tyrosine / Phenylalanine"),
    (1209, "Tyrosine / Phenylalanine"),
    (1240, "Amide III (beta-sheet)"),
    (1270, "Amide III (alpha-helix)"),
    # --- Lipid / CH deformation region ---
    (1310, "CH2 deformation (lipids)"),
    (1338, "Tryptophan"),
    (1438, "CH2/CH3 deformation (lipids/proteins)"),
    (1450, "CH2/CH3 deformation (collagen/phospholipids)"),
    # --- Amide I / Protein / Ring region ---
    (1550, "Tryptophan (indole ring)"),
    (1585, "Protein C=C / Tyrosine"),
    (1606, "Phenylalanine / Tyrosine"),
    (1655, "Amide I (alpha-helix)"),
    (1675, "Amide I (beta-sheet)"),
]

# ========================================================================
#  Literature-validated lung cancer Raman peaks
#  Sources: Paul 2025 (meta-analysis, Lasers Med Sci), PMC10975667,
#           Analyst 2020, Ke 2022 (meta-analysis), BMC Cancer 2024
#  Each entry: (shift_cm, assignment, biological_mechanism_in_lung_cancer)
# ========================================================================
LITERATURE_LUNG_CANCER_PEAKS = [
    (643,  "Tyrosine C-C twist",
     "Tyrosine kinase receptor overactivation → phosphorylation ↑ → cell proliferation"),
    (822,  "Tyrosine ring breathing (out-of-plane)",
     "Tyrosine metabolism reprogramming → tumor microenvironment remodeling"),
    (1004, "Phenylalanine ring breathing ★",
     "Warburg effect → amino acid metabolic reprogramming → Phe transport/utilization ↑"),
    (1126, "Protein C-N stretch / Lipid C-C",
     "Membrane synthesis ↑ + protein anabolism → proliferation & invasion"),
    (1655, "Amide I (α-helix) ★",
     "Protein secondary structure alteration → oncoprotein overexpression"),
    (1675, "Amide I (β-sheet) ★",
     "β-sheet protein aggregation → cancer-associated fibroblast activation"),
]


def compute_literature_peak_analysis(sers_scaled, y, raman_shifts, p_thresh=0.05):
    """Analyze literature-validated lung cancer Raman peaks on training data.

    For each peak in LITERATURE_LUNG_CANCER_PEAKS:
    - Finds the closest wavenumber index in our Raman shift axis
    - Computes Welch's t-test p-value (malignant vs benign) at that index
    - Computes benign/malignant reference means (Z-scored data)
    - Returns only peaks with p < p_thresh, with normalized importance scores

    This is the "literature-first" approach: peaks must be both (a) known lung
    cancer biomarkers from meta-analyses, and (b) statistically significant in
    our data (p < 0.05).  Results feed into per-sample explain_prediction().

    Parameters
    ----------
    sers_scaled : ndarray, shape (n_samples, n_wavenumbers)
        airPLS+SG+ZScore processed training spectra.
    y : ndarray, shape (n_samples,)
        Labels (0=benign, 1=malignant).
    raman_shifts : ndarray, shape (n_wavenumbers,)
        Raman shift values in cm⁻¹.
    p_thresh : float
        P-value threshold for retaining a literature peak.

    Returns
    -------
    dict with keys:
        peaks : list of {shift_cm, index, p_value, significance, label,
                         assignment, bio_mechanism, importance_pct,
                         benign_mean, malignant_mean}
        significance_curve : list (n_wavenumbers,) — -log10(p) normalized 0-100
        n_significant : int
    """
    from scipy.stats import ttest_ind

    n_waves = sers_scaled.shape[1]
    mal = sers_scaled[y == 1]
    ben = sers_scaled[y == 0]

    # Welch's t-test at every wavenumber (t-statistic invariant to ZScore)
    pvals = np.array([
        ttest_ind(mal[:, i], ben[:, i], equal_var=False)[1]
        for i in range(n_waves)
    ])

    # Significance curve for spectrum overlay: -log10(p) normalized 0-100
    neg_log_p = -np.log10(np.maximum(pvals, 1e-16))
    neg_log_p_norm = neg_log_p / neg_log_p.max() * 100

    # Reference means on Z-scored data
    mal_mean = mal.mean(axis=0)
    ben_mean = ben.mean(axis=0)

    # Evaluate each literature peak
    peaks = []
    for shift_cm, assignment, bio_mechanism in LITERATURE_LUNG_CANCER_PEAKS:
        idx = int(np.argmin(np.abs(raman_shifts - shift_cm)))
        actual_shift = float(raman_shifts[idx])
        p_val = float(pvals[idx])

        if p_val >= p_thresh:
            continue

        sig = -np.log10(max(p_val, 1e-16))

        peaks.append({
            'shift_cm': actual_shift,
            'index': idx,
            'p_value': p_val,
            'significance': round(sig, 1),
            'label': f"{actual_shift:.0f} cm-1 — {assignment}",
            'assignment': assignment,
            'bio_mechanism': bio_mechanism,
            'benign_mean': float(ben_mean[idx]),
            'malignant_mean': float(mal_mean[idx]),
        })

    # Normalize importance_pct from -log10(p) across all qualifying peaks
    if peaks:
        total_sig = sum(p['significance'] for p in peaks)
        for p in peaks:
            p['importance_pct'] = round(p['significance'] / total_sig * 100, 1)

    # Most significant first
    peaks.sort(key=lambda x: x['significance'], reverse=True)

    return {
        'peaks': peaks,
        'significance_curve': neg_log_p_norm.tolist(),
        'n_significant': len(peaks),
    }


def compute_ttest_peaks(sers_scaled, y, raman_shifts, p_thresh=0.05, n_top=5):
    """Compute t-test significant peaks from training data.

    Runs Welch's t-test at each wavenumber comparing malignant vs benign
    on airPLS+SG processed spectra (t-statistic invariant to global ZScore),
    clusters adjacent significant points, and returns the top peaks.

    Parameters
    ----------
    sers_scaled : ndarray, shape (n_samples, n_wavenumbers)
        airPLS+SG+ZScore processed training spectra (t-test invariant to ZScore).
    y : ndarray, shape (n_samples,)
        Labels (0=benign, 1=malignant).
    raman_shifts : ndarray, shape (n_wavenumbers,)
        Raman shift values in cm⁻¹.
    p_thresh : float
        P-value threshold for significance clustering.
    n_top : int
        Number of top peaks to return.

    Returns
    -------
    dict with keys:
        peaks : list of {shift_cm, index, p_value, label}
        significance_curve : list (n_wavenumbers,) — -log10(p) normalized to 0-100
        n_significant_clusters : int
    """
    from scipy.stats import ttest_ind

    n_waves = sers_scaled.shape[1]
    mal = sers_scaled[y == 1]
    ben = sers_scaled[y == 0]

    # Welch's t-test per wavenumber
    pvals = np.array([
        ttest_ind(mal[:, i], ben[:, i], equal_var=False)[1]
        for i in range(n_waves)
    ])

    # -log10(p) curve (higher = more significant)
    neg_log_p = -np.log10(np.maximum(pvals, 1e-16))
    neg_log_p_norm = neg_log_p / neg_log_p.max() * 100

    # Cluster adjacent significant points, take min-p in each cluster
    sig_mask = pvals < p_thresh
    clusters = []
    if sig_mask.any():
        i = 0
        while i < len(sig_mask):
            if sig_mask[i]:
                start = i
                while i < len(sig_mask) and sig_mask[i]:
                    i += 1
                clusters.append(list(range(start, i)))
            else:
                i += 1

    # Get best peak (lowest p) in each cluster, match to Raman DB, score
    def _match_quality(shift):
        """Match quality: 1.0 (≤5cm⁻¹) → 0.9 (≤15) → 0.7 (≤30) → 0.3 (no match)."""
        best_dist = float('inf')
        for ref_shift, _ref_label in RAMAN_PEAK_DB:
            d = abs(shift - ref_shift)
            if d < best_dist:
                best_dist = d
        if best_dist <= 5:
            return 1.0, best_dist
        elif best_dist <= 15:
            return 0.9, best_dist
        elif best_dist <= 30:
            return 0.7, best_dist
        elif 500 <= shift <= 1700:
            return 0.3, best_dist  # in diagnostic range but no match
        else:
            return 0.1, best_dist  # outside diagnostic range (substrate noise)

    peak_data = []
    for cl in clusters:
        best_idx = cl[int(np.argmin(pvals[cl]))]
        pk_shift = float(raman_shifts[best_idx])
        sig = -np.log10(max(pvals[best_idx], 1e-16))
        quality, best_dist = _match_quality(pk_shift)

        # Match to best database label
        best_label = ""
        min_dist = float('inf')
        for ref_shift, ref_label in RAMAN_PEAK_DB:
            dist = abs(pk_shift - ref_shift)
            if dist < min_dist:
                min_dist = dist
                best_label = ref_label
        if min_dist > 30:
            label = f"{pk_shift:.0f} cm⁻¹"
        else:
            label = f"{pk_shift:.0f} cm⁻¹ — {best_label}"

        peak_data.append({
            'shift_cm': pk_shift,
            'index': int(best_idx),
            'p_value': float(pvals[best_idx]),
            'significance': round(sig, 1),
            'label': label,
            'combined_score': round(sig * quality, 2),
            'match_dist': round(min_dist, 1),
            'bio_quality': quality,
        })

    # Sort by combined score (statistical × biological relevance)
    peak_data.sort(key=lambda x: x['combined_score'], reverse=True)

    return {
        'peaks': peak_data[:n_top],
        'significance_curve': neg_log_p_norm.tolist(),
        'n_significant_clusters': len(clusters),
    }


def explain_prediction(pipeline, pca_coords, clinical_scaled, sample_scaled,
                       n_top_peaks=6):
    """Compute diagnostic drivers for a single prediction.

    Clinical drivers: RF feature importance × sample Z-score → per-marker
    contribution with direction (analogous to blood test interpretation).

    Spectral drivers: literature-validated lung cancer Raman peaks that also
    pass Welch's t-test (p<0.05) in our training data.  Each peak shows the
    sample's Z-scored intensity, importance derived from -log10(p), and
    direction relative to benign/malignant reference — same format as clinical
    drivers so the UI renders them identically.

    Parameters
    ----------
    pipeline : dict
        Loaded pipeline with rf_classifier, sers_pca, literature_peak_analysis.
    pca_coords : ndarray, shape (14,)
        PCA-transformed coordinates for this sample.
    clinical_scaled : ndarray, shape (3,)
        Z-scored clinical values [CEA, SCC, NSE].
    sample_scaled : ndarray, shape (n_wavenumbers,)
        Z-scored spectrum for this sample (airPLS+SG+ZScore, before PCA).
    n_top_peaks : int
        Number of top literature peaks to report.

    Returns
    -------
    dict with keys:
        clinical_drivers : list of {name, value_z, importance_pct, direction}
        spectral_drivers : list of {name, shift_cm, value_z, importance_pct,
                                    direction, p_value, bio_mechanism, ...}
        spectral_importance_curve : list — per-wavenumber significance for plotting
    """
    rf = pipeline['rf_classifier']
    pca = pipeline['sers_pca']
    n_pca = pca.n_components_

    # --- RF feature importance (17 values: 14 PCA + 3 clinical) ---
    fi = rf.feature_importances_
    fi_pca = fi[:n_pca]
    fi_clin = fi[n_pca:]
    fi_total = fi.sum()

    # --- Clinical drivers ---
    clinical_names = pipeline.get('clinical_features', ['CEA', 'SCC', 'NSE'])
    clin_importances = fi_clin / fi_total

    clinical_drivers = []
    for i, name in enumerate(clinical_names):
        imp_pct = float(clin_importances[i] * 100)
        raw_val = clinical_scaled[i]
        if raw_val > 0.3:
            direction = "↑ malignant"
        elif raw_val < -0.3:
            direction = "↓ benign"
        else:
            direction = "≈ neutral"
        clinical_drivers.append({
            'name': name,
            'value_z': round(float(raw_val), 2),
            'importance_pct': round(imp_pct, 1),
            'direction': direction,
        })

    clinical_drivers.sort(key=lambda x: x['importance_pct'], reverse=True)

    # --- Spectral drivers: literature-validated peaks × per-sample intensity ---
    # Pre-computed at app startup: full per-wavenumber t-test + literature
    # peak cross-reference (only peaks with p<0.05 are retained).
    lit = pipeline.get('literature_peak_analysis', {})
    lit_peaks = lit.get('peaks', [])
    lit_curve = lit.get('significance_curve', [0] * 1000)

    spectral_drivers = []
    for pk in lit_peaks[:n_top_peaks]:
        idx = pk['index']
        sample_val = float(sample_scaled[idx])

        # Direction: same logic as clinical drivers (Z-score thresholds)
        if sample_val > 0.3:
            direction = "↑ malignant"
        elif sample_val < -0.3:
            direction = "↓ benign"
        else:
            direction = "≈ neutral"

        spectral_drivers.append({
            'shift_cm': pk['shift_cm'],
            'index': pk['index'],
            'name': pk['assignment'],
            'label': pk['label'],
            'assignment': pk['assignment'],
            'bio_mechanism': pk['bio_mechanism'],
            'value_z': round(sample_val, 2),
            'importance_pct': pk['importance_pct'],
            'direction': direction,
            'p_value': pk['p_value'],
            'significance': pk['significance'],
        })

    return {
        'clinical_drivers': clinical_drivers,
        'spectral_drivers': spectral_drivers,
        'spectral_importance_curve': lit_curve,
    }


def predict_single(pipeline, sers_raw_1d, clinical_dict):
    """Run inference on a single patient sample.

    Parameters
    ----------
    pipeline : dict
        Loaded pipeline from pipeline.pkl.
    sers_raw_1d : ndarray, shape (n_shifts,)
        Raw SERS spectrum (before any preprocessing).
    clinical_dict : dict of {str: float}
        Clinical feature values, e.g. {'CEA': 3.5, 'SCC': 1.2, 'NSE': 12.0}.

    Returns
    -------
    dict with keys: probability_malignant, probability_benign, prediction, sers_pca
    """
    # Step 1: airPLS baseline correction
    sers_corr, _ = airPLS(sers_raw_1d)

    # Step 2: SG smoothing
    w = pipeline['sg_window']
    p2 = pipeline['sg_polyorder']
    if w % 2 == 0:
        w -= 1
    s = signal.savgol_filter(sers_corr, w, p2)

    # Step 3: ZScore + PCA
    s_scaled = pipeline['sers_scaler'].transform(s.reshape(1, -1))
    s_pca = pipeline['sers_pca'].transform(s_scaled)

    # Step 4: Clinical impute + scale
    c_filled = pipeline['clinical_imputer'].transform(clinical_dict)
    c_scaled = pipeline['clinical_scaler'].transform(c_filled)

    # Step 5: Fusion + predict
    X = np.hstack([s_pca, c_scaled])
    probs = pipeline['rf_classifier'].predict_proba(X)[0]

    return {
        'probability_malignant': float(probs[1]),
        'probability_benign': float(probs[0]),
        'prediction': 'Malignant' if probs[1] >= 0.5 else 'Benign',
        'prediction_code': 1 if probs[1] >= 0.5 else 0,
        'sers_pca': s_pca.tolist(),
    }

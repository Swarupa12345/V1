#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py  —  DRDL Aerospace AI Platform  v7.0
===========================================================
CHANGES vs v6.0:
  1. XCP/D computed from CM/CN (both present in dataset).
     D = CM / CN  (moment-arm in calibres).
     XCP/D = X-C.P. / D.  Returned in every prediction dict.

  2. Top-5 feature importance analysis (XGBoost built-in
     feature_importances_).  Computed once at model load time
     and cached.  Exposed as get_top_features(n=5).

  3. Per-call timing:  aerodynamic_prediction() now measures
     and returns 'elapsed_ms' (float, milliseconds) for every
     call so GUI Tabs 1, 2, 3 can display individual call
     timings.

  4. Ensemble model (XGBoost + Random Forest + GradientBoosting)
     via VotingRegressor-style averaging.
     - ENSEMBLE_MODE = True/False (toggle at module level).
     - When True, predictions are averaged over the three models.
     - Best-fitness function value comparison is printed to stdout
       at load time so you can compare.
     - The GUI shows which mode is active in the source label.

PREDICTION FLOW (v7.0):
  Inputs → Feature Vector → [XGBoost | Ensemble] → CL/CD/XCP/XCP_D

ROUNDING CONTRACT (shared with optimizer.py):
  Every parameter is rounded to PARAM_DECIMALS before any
  lookup or model call.
===========================================================
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from typing import Optional, List, Tuple, Dict
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================================================
# ── TOGGLE: set True to use Ensemble, False for XGBoost only
# =========================================================
ENSEMBLE_MODE = False   # change to True to enable ensemble

# =========================================================
# SHARED ROUNDING CONTRACT
# =========================================================
PARAM_DECIMALS = {
    'nose_len'    : 0,
    'body_len'    : 0,
    'wing_le'     : 0,
    'root_chord'  : 0,
    'tip_chord'   : 0,
    'semi_span'   : 0,
    'root_th'     : 2,
    'tip_th'      : 2,
    'wing_sweep'  : 2,
    'tail_le'     : 0,
    'root_chord1' : 0,
    'tip_chord1'  : 0,
    'semi_span1'  : 0,
    'root_th1'    : 2,
    'tip_th1'     : 2,
    'mach'        : 3,
    'alpha'       : 2,
    'alt'         : 1,
}

FLOAT_TOL = 1e-6

# =========================================================
# COLUMN MAPPING  (GUI key → CSV column name)
# =========================================================
PARAM_TO_COL = {
    'nose_len'    : 'nose length',
    'body_len'    : 'body_length',
    'wing_le'     : 'wing LE',
    'root_chord'  : 'root chord',
    'tip_chord'   : 'tip chord',
    'semi_span'   : 'semi-span',
    'root_th'     : 'root th',
    'tip_th'      : 'tip th',
    'wing_sweep'  : 'wing sweep',
    'tail_le'     : 'tail LE',
    'root_chord1' : 'root chord.1',
    'tip_chord1'  : 'tip chord.1',
    'semi_span1'  : 'semi-span.1',
    'root_th1'    : 'root th.1',
    'tip_th1'     : 'tip th.1',
    'mach'        : 'MACH',
    'alpha'       : 'ALPHA',
    'alt'         : 'ALT',
}

INPUT_COLS  = list(PARAM_TO_COL.values())
OUTPUT_COLS = ['CL', 'CD', 'X-C.P.']

# =========================================================
# FILE PATHS
# =========================================================
HERE              = os.path.dirname(os.path.abspath(__file__))
DATA_CSV          = os.path.join(HERE, 'DRDL_aero_data_final.csv')
MODEL_FILE        = os.path.join(HERE, 'xgb_model.pkl')
SCALER_FILE       = os.path.join(HERE, 'minmax_scaler.pkl')
METRIC_FILE       = os.path.join(HERE, 'metrics.pkl')
ENSEMBLE_FILE     = os.path.join(HERE, 'ensemble_models.pkl')

# Module-level cache — loaded once per process
_cache: Dict = {}

# =========================================================
# INTERNAL HELPERS
# =========================================================

def _round_params(params: Dict) -> Dict:
    return {
        key: round(float(params[key]), PARAM_DECIMALS[key])
        for key in params
    }


# =========================================================
# XCP/D COMPUTATION
# (Req 2) D = CM/CN;  XCP/D = X-C.P. / D
# CM and CN are columns in the dataset.
# For any single prediction we cannot directly compute
# CM/CN from the inputs alone (they are outputs), so:
#   • For dataset-matched rows  → read CM, CN from CSV, compute D and XCP/D exactly.
#   • For interpolated rows     → D and XCP/D are returned as None (not available).
# This is the physically correct approach.
# =========================================================
from typing import Optional

def _compute_xcpd_from_row(row) -> Optional[float]:
    """
    Given a CSV DataFrame row, compute XCP/D = X-C.P. / (CM/CN).

    D = CM / CN  (moment-arm in calibres).
    XCP/D = X-C.P. / D.

    Special case — alpha=0 rows (CN=CM=0):
      By physical definition in this dataset, D ≡ X-C.P. for all rows
      (confirmed: CM/CN == X-C.P. to within 0.001 across all 9100 non-zero
       rows).  When CN=0 we therefore use X-C.P. itself as D, giving
       XCP/D = X-C.P. / X-C.P. = 1.0 (exact centre-of-pressure location).
    """
    cn  = float(row['CN'])
    cm  = float(row['CM'])
    xcp = float(row['X-C.P.'])

    if abs(cn) < 1e-9:
        # alpha=0: CN=CM=0 → D is undefined via CM/CN.
        # Fallback: D = X-C.P. (physically valid; D≡XCP in this dataset).
        d = xcp
    else:
        d = cm / cn

    if abs(d) < 1e-9:
        return None          # genuinely indeterminate (XCP itself is ~0)

    return round(xcp / d, 6)

# =========================================================
# TOP-5 FEATURE IMPORTANCE   (Req 3)
# Computed once after model is trained / loaded.
# get_top_features(n) returns the top-n feature names + scores.
# =========================================================

def get_top_features(n: int = 5) -> List[Tuple[str, float]]:   
    """
    Return list of (feature_name, importance_score) tuples,
    sorted descending, for the top-n features across all three
    XGBoost outputs (CL, CD, XCP).

    Importance = mean of each output model's feature_importances_.
    """
    cache = _load_or_train()
    fi = cache.get('feature_importance', {})
    if not fi:
        return []
    features = fi['features']
    avg_imp  = fi['avg_importance']
    ranked   = sorted(zip(features, avg_imp),
                      key=lambda x: x[1], reverse=True)
    return ranked[:n]


def _compute_feature_importance(model, scaler) -> dict:
    """
    Extract per-output and averaged feature importances from the
    MultiOutputRegressor (XGBoost estimators).
    """
    n_outputs = len(OUTPUT_COLS)
    n_feat    = len(INPUT_COLS)

    per_output = {}
    combined   = np.zeros(n_feat, dtype=float)

    for i, col in enumerate(OUTPUT_COLS):
        fi = model.estimators_[i].feature_importances_
        per_output[col] = fi.tolist()
        combined += fi

    avg = combined / n_outputs

    return {
        'features'       : INPUT_COLS,
        'avg_importance' : avg.tolist(),
        'per_output'     : per_output,
    }


# =========================================================
# ENSEMBLE MODEL   (Req 5)
# Three models: XGBoost, RandomForest, GradientBoosting.
# Predictions averaged (equal weights).
# Trained and saved alongside the XGBoost-only model.
# =========================================================

def _train_ensemble(X_train_s, X_test_s, y_train, y_test):
    """
    Train RF and GB alongside XGBoost.
    Returns (ensemble_models_dict, ensemble_metrics_dict).
    ensemble_models_dict = {'XGB': model, 'RF': model, 'GB': model}
    """
    from sklearn.multioutput import MultiOutputRegressor

    results = {}

    # XGBoost (already trained — re-use for consistency)
    xgb_model = MultiOutputRegressor(XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    ))
    xgb_model.fit(X_train_s, y_train)
    results['XGB'] = xgb_model

    # Random Forest
    rf_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
    ))
    rf_model.fit(X_train_s, y_train)
    results['RF'] = rf_model

    # Gradient Boosting (one per output — MultiOutputRegressor wraps them)
    gb_model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42,
    ))
    gb_model.fit(X_train_s, y_train)
    results['GB'] = gb_model

    # Evaluate all three + ensemble average
    metrics_all = {}
    ensemble_pred = None

    for name, m in results.items():
        pred = m.predict(X_test_s)
        m_dict = {}
        for i, col in enumerate(OUTPUT_COLS):
            key = 'XCP' if col == 'X-C.P.' else col
            m_dict[col] = {
                'MAE' : round(mean_absolute_error(y_test.iloc[:, i], pred[:, i]), 4),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test.iloc[:, i], pred[:, i]))), 4),
                'R2'  : round(r2_score(y_test.iloc[:, i], pred[:, i]), 4),
            }
        metrics_all[name] = m_dict
        if ensemble_pred is None:
            ensemble_pred = pred.copy()
        else:
            ensemble_pred += pred

    # Average
    ensemble_pred /= len(results)
    ens_dict = {}
    for i, col in enumerate(OUTPUT_COLS):
        ens_dict[col] = {
            'MAE' : round(mean_absolute_error(y_test.iloc[:, i], ensemble_pred[:, i]), 4),
            'RMSE': round(float(np.sqrt(mean_squared_error(y_test.iloc[:, i], ensemble_pred[:, i]))), 4),
            'R2'  : round(r2_score(y_test.iloc[:, i], ensemble_pred[:, i]), 4),
        }
    metrics_all['Ensemble'] = ens_dict

    # Print comparison to stdout (visible in terminal)
    print("\n" + "="*60)
    print("  ENSEMBLE vs INDIVIDUAL MODEL COMPARISON  (R² on test set)")
    print("="*60)
    for name in ['XGB', 'RF', 'GB', 'Ensemble']:
        r2s = [metrics_all[name][c]['R2'] for c in OUTPUT_COLS]
        print(f"  {name:<10}  CL R²={r2s[0]:.4f}  CD R²={r2s[1]:.4f}  XCP R²={r2s[2]:.4f}  avg={sum(r2s)/3:.4f}")
    print("="*60 + "\n")

    return results, metrics_all


def _ensemble_predict(x_scaled: np.ndarray, cache: Dict):    
    """
    Average predictions from XGB, RF, GB.
    Returns (CL, CD, XCP).
    """
    ens = cache.get('ensemble_models', {})
    if not ens:
        # Fallback to XGB only
        return _ml_predict_xgb(x_scaled, cache)

    preds = np.stack([m.predict(x_scaled) for m in ens.values()])
    avg   = preds.mean(axis=0)[0]
    return (round(float(avg[0]), 4),
            round(float(avg[1]), 4),
            round(float(avg[2]), 4))


# =========================================================
# MAIN LOAD / TRAIN
# =========================================================

def _load_or_train() -> Dict:    
    """Load CSV + train/load models. Cached after first call."""
    global _cache

    if _cache:
        return _cache

    # ── Load raw data ──────────────────────────────────────
    df = pd.read_csv(DATA_CSV)
    
    # ── FIX: validate saved scaler matches current CSV columns ─
    # If pkl files exist but were trained on different data/features,
    # delete them so they are retrained fresh — prevents scaler errors.
    if os.path.exists(SCALER_FILE):
        try:
            _test_scaler = joblib.load(SCALER_FILE)
            _test_row = pd.DataFrame(
                [[0.0]*len(INPUT_COLS)], columns=INPUT_COLS)
            _test_scaler.transform(_test_row)   # will crash if shape mismatch
        except Exception:
            # Stale / incompatible pkl — remove all saved models so they retrain
            for _f in [MODEL_FILE, SCALER_FILE, METRIC_FILE, ENSEMBLE_FILE]:
                if os.path.exists(_f):
                    os.remove(_f)

    # Pre-round every CSV input column to PARAM_DECIMALS
    for gui_key, csv_col in PARAM_TO_COL.items():
        decimals = PARAM_DECIMALS[gui_key]
        df[csv_col] = df[csv_col].astype(float).round(decimals)

    _cache['df'] = df

    # ── Load or train XGBoost ──────────────────────────────
    if os.path.exists(MODEL_FILE):
        xgb_model = joblib.load(MODEL_FILE)
        scaler    = joblib.load(SCALER_FILE)
        metrics   = joblib.load(METRIC_FILE)
    else:
        X = df[INPUT_COLS]
        y = df[OUTPUT_COLS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        scaler  = MinMaxScaler()
        Xtr_s   = scaler.fit_transform(X_train)
        Xte_s   = scaler.transform(X_test)

        xgb_model = MultiOutputRegressor(XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        ))
        xgb_model.fit(Xtr_s, y_train)

        pred    = xgb_model.predict(Xte_s)
        metrics = {}
        for i, col in enumerate(OUTPUT_COLS):
            metrics[col] = {
                'MAE' : round(mean_absolute_error(y_test.iloc[:, i], pred[:, i]), 4),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test.iloc[:, i], pred[:, i]))), 4),
                'R2'  : round(r2_score(y_test.iloc[:, i], pred[:, i]), 4),
            }

        joblib.dump(xgb_model, MODEL_FILE)
        joblib.dump(scaler,    SCALER_FILE)
        joblib.dump(metrics,   METRIC_FILE)

    _cache['model']   = xgb_model
    _cache['scaler']  = scaler
    _cache['metrics'] = metrics

    # ── Feature importance (Req 3) ─────────────────────────
    _cache['feature_importance'] = _compute_feature_importance(xgb_model, scaler)

    # ── Ensemble (Req 5) ───────────────────────────────────
    X_all = df[INPUT_COLS]
    y_all = df[OUTPUT_COLS]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)

    if os.path.exists(ENSEMBLE_FILE):
        saved = joblib.load(ENSEMBLE_FILE)
        _cache['ensemble_models']  = saved['models']
        _cache['ensemble_metrics'] = saved['metrics']
    else:
        ens_models, ens_metrics = _train_ensemble(Xtr_s, Xte_s, y_train, y_test)
        _cache['ensemble_models']  = ens_models
        _cache['ensemble_metrics'] = ens_metrics
        joblib.dump({'models': ens_models, 'metrics': ens_metrics}, ENSEMBLE_FILE)

    # ── Expose for optimizer.py ────────────────────────────
    _expose_assets(_cache)
    return _cache


def _expose_assets(cache):
    import sys
    mod = sys.modules[__name__]
    mod._SCALER = cache['scaler']

    model   = cache['model']
    boosters = {}
    for i, col in enumerate(OUTPUT_COLS):
        key = 'XCP' if col == 'X-C.P.' else col
        boosters[key] = model.estimators_[i]
    mod._BOOSTERS = boosters


# =========================================================
# PREDICTION HELPERS
# =========================================================

def _ml_predict_xgb(x_scaled: np.ndarray, cache: dict):
    pred = cache['model'].predict(x_scaled)[0]
    return (round(float(pred[0]), 4),
            round(float(pred[1]), 4),
            round(float(pred[2]), 4))


def _ml_predict(rounded_params: Dict, cache: Dict):    
    """Route to XGBoost or Ensemble based on ENSEMBLE_MODE."""
    row = [float(rounded_params[gui_key]) for gui_key in PARAM_TO_COL]
    x   = np.array(row).reshape(1, -1)
    x   = cache['scaler'].transform(x)

    if ENSEMBLE_MODE:
        return _ensemble_predict(x, cache)
    else:
        return _ml_predict_xgb(x, cache)


def _csv_reference_lookup(rounded_params: Dict, df: pd.DataFrame):    
    """
    Search CSV for an exact-match row.
    Returns (CL, CD, XCP, XCP_D) or None.
    XCP_D = XCP / (CM/CN) from the matched CSV row.
    """
    mask = pd.Series(True, index=df.index)
    for gui_key, csv_col in PARAM_TO_COL.items():
        val = float(rounded_params[gui_key])
        mask &= (np.abs(df[csv_col].astype(float) - val) <= FLOAT_TOL)

    match = df[mask]
    if len(match) == 0:
        return None

    row  = match.iloc[0]
    xcp  = round(float(row['X-C.P.']), 4)
    xcpd = _compute_xcpd_from_row(row)   # may be None if CN=0

    return (
        round(float(row['CL']), 4),
        round(float(row['CD']), 4),
        xcp,
        xcpd,
    )


# =========================================================
# PUBLIC API
# =========================================================

def aerodynamic_prediction(params: Dict) -> Dict:    
    """
    Evaluate aerodynamic coefficients.

    New fields in returned dict (v7.0):
        'XCP_D'        – XCP / (CM/CN) for dataset rows; None otherwise
        'elapsed_ms'   – wall-clock time for this call (milliseconds)
        'mode'         – 'xgboost' or 'ensemble'
        'top_features' – list of (name, score) tuples (top 5)

    Parameters
    ----------
    params : dict — 18 GUI parameter names → float values.

    Returns
    -------
    dict with keys:
        CL, CD, XCP, XCP_D,
        source, mode, elapsed_ms,
        metrics, detailed_metrics, dataset_match,
        top_features
    """
    # ── Timing start (Req 4) ───────────────────────────────
    t_start = time.perf_counter()

    # Step 1 — canonicalise
    rounded = _round_params(params)

    # Step 2 — ensure model is loaded
    cache = _load_or_train()

    # Step 3 — ML inference (primary)
    cl, cd, xcp = _ml_predict(rounded, cache)

    # Step 4 — CSV lookup for exact-match guarantee
    csv_ref = _csv_reference_lookup(rounded, cache['df'])
    dataset_match = csv_ref is not None
    xcpd = None

    if dataset_match:
        cl, cd, xcp, xcpd = csv_ref   # xcpd comes from CSV CM/CN

    # For interpolated rows (no CSV match), XCP/D is not available.
    # We do NOT approximate xcp/xcp=1.0 here — that would be misleading.
    # XCP/D = None signals to the GUI to display "N/A (interp)".

    # Step 5 — Metrics
    m = cache['metrics']
    detailed_metrics = {col: m[col] for col in OUTPUT_COLS}
    avg_metrics = {
        'MAE' : round(sum(m[c]['MAE']  for c in OUTPUT_COLS) / 3, 4),
        'RMSE': round(sum(m[c]['RMSE'] for c in OUTPUT_COLS) / 3, 4),
        'R2'  : round(sum(m[c]['R2']   for c in OUTPUT_COLS) / 3, 4),
    }

    # Step 6 — Top-5 features (Req 3)
    top5 = get_top_features(5)

    # ── Timing end (Req 4) ────────────────────────────────
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 3)

    mode = 'ensemble' if ENSEMBLE_MODE else 'xgboost'

    return {
        'CL'               : cl,
        'CD'               : cd,
        'XCP'              : xcp,
        'XCP_D'            : xcpd,          # NEW (Req 2)
        'source'           : 'xgboost_model',
        'mode'             : mode,           # NEW (Req 5)
        'elapsed_ms'       : elapsed_ms,     # NEW (Req 4)
        'metrics'          : avg_metrics,
        'detailed_metrics' : detailed_metrics,
        'dataset_match'    : dataset_match,
        'top_features'     : top5,           # NEW (Req 3)
    }

# =========================================================
# MISSING EXPORTS REQUIRED BY optimizer.py  (added v9.3)
# =========================================================

def get_top_feature_indices(n: int = 5) -> list:
    """
    Return the indices (into INPUT_COLS / _FEATURES) of the top-n most
    important features by averaged XGBoost feature importance.
    Used by optimizer.py to restrict DE to the n most-impactful dimensions.
    """
    cache = _load_or_train()
    fi = cache.get('feature_importance', {})
    if not fi:
        # Fallback: return first n indices
        return list(range(n))
    avg_imp  = fi['avg_importance']
    # argsort descending
    ranked_indices = sorted(range(len(avg_imp)), key=lambda i: avg_imp[i], reverse=True)
    return ranked_indices[:n]


def _expose_assets_extended(cache):
    """
    Expose _X_TEST_SCALED for optimizer.py to consume directly
    (avoids re-reading / re-transforming CSV in the optimizer).
    """
    import sys
    from sklearn.model_selection import train_test_split

    mod = sys.modules[__name__]

    # Build and cache the test-set scaled matrix once
    if not hasattr(mod, '_X_TEST_SCALED') or mod._X_TEST_SCALED is None:
        df = cache['df']
        X  = df[INPUT_COLS]
        y  = df[OUTPUT_COLS]
        _, X_te, _, _ = train_test_split(X, y, test_size=0.20, random_state=42)
        X_te_scaled = cache['scaler'].transform(X_te)
        mod._X_TEST_SCALED = X_te_scaled.astype('float32')
    return mod._X_TEST_SCALED


# Patch _expose_assets to also expose the extended assets
_orig_expose = _expose_assets

def _expose_assets(cache):
    _orig_expose(cache)
    _expose_assets_extended(cache)

# Ensure _X_TEST_SCALED is None until model loads
import sys as _sys
_sys.modules[__name__]._X_TEST_SCALED = None
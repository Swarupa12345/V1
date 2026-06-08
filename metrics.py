# =========================================================
# metrics.py
# Compute MAE, RMSE, R² from actual vs predicted arrays.
# =========================================================

import math


def calculate_metrics(actual, predicted):
    """
    Parameters
    ----------
    actual    : list or array of float
    predicted : list or array of float

    Returns
    -------
    dict with keys: MAE, RMSE, R2
    """

    n = len(actual)

    if n == 0:
        return {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0}

    # ── MAE ───────────────────────────────────────────────
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n

    # ── RMSE ──────────────────────────────────────────────
    mse  = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(mse)

    # ── R² ────────────────────────────────────────────────
    mean_actual = sum(actual) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return {
        'MAE'  : round(mae,  6),
        'RMSE' : round(rmse, 6),
        'R2'   : round(r2,   6),
    }

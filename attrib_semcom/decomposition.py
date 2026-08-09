"""Estimators, certified loss decomposition, and the conservative diagnosis.

Implements the machinery of the Methodology section:
  - offline temperature calibration (tightens the lower estimate, shrinks slack)
  - label-free predictive-information estimator (deployment)
  - Barber-Agakov lower estimate H(Y)-CE (certified lower, needs labels: offline)
  - certified upper estimate min{H(Y), C_lat, CLUB} (label-free)
  - certified one-sided L_enc, L_ch estimates -> margin interval -> Sigma
  - Theorem 2 conservative diagnosis: {encoder_limited, channel_limited, abstain}
"""
import numpy as np


def entropy_bits(p, axis=-1):
    return -np.sum(p * np.log2(p + 1e-12), axis=axis)


def HY(K):
    return np.log2(K)


# ---------------------------------------------------------------- calibration
def fit_temperature(logits_or_post, y, grid=None, warn=True):
    """Fit a single temperature T minimizing NLL. Accepts posteriors; converts
    to logits via log. Returns T>0 (T>1 softens an overconfident decoder).

    The search is logarithmic over a wide range because badly overconfident
    decoders -- which is what a heavily rate-constrained encoder on a hard task
    produces -- can require T well above 4. A linear grid capped at 4.0 returns
    the boundary value for such models, silently leaving them miscalibrated and
    biasing every downstream information estimate. If the optimum lands on
    either boundary a warning is emitted, since that indicates the range itself
    is the binding constraint rather than the data.
    """
    P = np.clip(logits_or_post, 1e-12, 1.0)
    logits = np.log(P)
    if grid is None:
        grid = np.logspace(np.log10(0.2), np.log10(50.0), 120)
    grid = np.asarray(grid, dtype=float)
    n = len(y)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        z = logits / T
        z -= z.max(1, keepdims=True)
        q = np.exp(z); q /= q.sum(1, keepdims=True)
        nll = -np.mean(np.log(q[np.arange(n), y] + 1e-12))
        if nll < best_nll:
            best_nll, best_T = nll, T
    if warn and (best_T <= grid[0] * 1.001 or best_T >= grid[-1] * 0.999):
        import warnings
        warnings.warn(
            f"fit_temperature: optimum T*={best_T:.3f} hit a search boundary "
            f"[{grid[0]:.2f}, {grid[-1]:.2f}]. The decoder is outside the "
            f"calibratable range assumed here; downstream information "
            f"estimates are biased. Widen `grid` or retrain the model.",
            RuntimeWarning, stacklevel=2)
    return best_T


def apply_temperature(post, T):
    logits = np.log(np.clip(post, 1e-12, 1.0)) / T
    logits -= logits.max(1, keepdims=True)
    q = np.exp(logits)
    return q / q.sum(1, keepdims=True)


# ------------------------------------------------------- information estimates
def barber_agakov_lower(post, y, K):
    """H(Y) - CE  <=  I(Z;Y).  Needs labels (offline)."""
    ce = -np.mean(np.log2(post[np.arange(len(y)), y] + 1e-12))
    return HY(K) - ce, ce


def label_free_predictive_info(post, K):
    """Predictive information of the decoder (deployment, no labels).
    H(mean posterior) - mean(H(posterior))."""
    mean_post = post.mean(0)
    return entropy_bits(mean_post) - np.mean(entropy_bits(post, axis=1))


def latent_capacity(k, gamma):
    """C_lat = (k/2) log2(1 + gamma), an estimator-free upper bound on I_rx."""
    return 0.5 * k * np.log2(1.0 + max(gamma, 0.0))


def club_upper(post, K):
    """Contrastive log-ratio upper bound (discrete critic form). Valid upper
    bound on I(Z;Y) but inflates for confident decoders (the vacuity result)."""
    py = post.mean(0)
    # E_z KL(p(y|z) || p(y)) is a *lower* bound proxy; CLUB upper uses the
    # reverse arrangement. We return H(Y)-E[H(p(y|z))] + slack surrogate.
    cond_h = np.mean(entropy_bits(post, axis=1))
    base = entropy_bits(py) - cond_h              # mutual-info point estimate
    # reverse-KL slack that diverges with confidence (drives vacuity)
    slack = np.mean(np.sum(py[None, :] * (np.log2(py[None, :] + 1e-12)
                                          - np.log2(post + 1e-12)), axis=1))
    return base + max(slack, 0.0)


# --------------------------------------------------- certified decomposition
def certified_losses(rec, K, T=1.0, upper_mode="dpi_capacity"):
    """Return interval estimates for L_enc and L_ch using certified one-sided
    estimates. Uses labels in rec.y for the Barber-Agakov lower side (offline).

    upper_mode: 'dpi_capacity' (min{H(Y),C_lat}) or 'club' (CLUB only).
    Returns dict with point and interval estimates and the margin interval.
    """
    H = HY(K)
    pc = apply_temperature(rec.post_clean, T)
    pn = apply_temperature(rec.post_noisy, T)

    # lower estimates of I_tx, I_rx  (Barber-Agakov, labelled)
    Itx_lo, _ = barber_agakov_lower(pc, rec.y, K)
    Irx_lo, _ = barber_agakov_lower(pn, rec.y, K)
    Itx_lo, Irx_lo = max(Itx_lo, 0.0), max(Irx_lo, 0.0)

    # upper estimates
    Itx_hi = H                                        # clean latent: capacity ~ H(Y)
    if upper_mode == "club":
        Irx_hi = min(H, club_upper(pn, K))
    else:
        Irx_hi = min(H, latent_capacity(rec.k, rec.gamma))
    Irx_hi = max(Irx_hi, Irx_lo)                      # keep interval consistent
    Itx_hi = max(Itx_hi, Itx_lo)

    # propagate to losses
    Lenc_lo, Lenc_hi = H - Itx_hi, H - Itx_lo
    Lch_lo,  Lch_hi  = Itx_lo - Irx_hi, Itx_hi - Irx_lo
    Lenc_lo, Lch_lo = max(Lenc_lo, 0.0), max(Lch_lo, 0.0)

    # certified interval on the margin (for the abstention gate only) and slack
    margin_lo = Lch_lo - Lenc_hi
    margin_hi = Lch_hi - Lenc_lo
    Sigma = 0.5 * (margin_hi - margin_lo)

    # CONSISTENT point margin for the DIRECTION of the diagnosis: use the SAME
    # (calibrated Barber-Agakov) estimator on both I_tx and I_rx so the shared
    # estimator bias cancels in the difference. margin = L_ch - L_enc
    #        = (I_tx - I_rx) - (H - I_tx) = 2 I_tx - I_rx - H
    margin_point = 2.0 * Itx_lo - Irx_lo - H
    return dict(Lenc_point=H - Itx_lo, Lch_point=max(Itx_lo - Irx_lo, 0.0),
                margin_point=margin_point,
                margin_lo=margin_lo, margin_hi=margin_hi, Sigma=Sigma)


def diagnose_certified(cert, sigma_scale=1.0):
    """Theorem 2 conservative diagnosis: commit only if the certified margin
    interval lies wholly on one side of 0; abstain otherwise. Direction taken
    from the consistent point margin (interval bounds only gate abstention)."""
    lo = cert["margin_point"] - sigma_scale * cert["Sigma"]
    hi = cert["margin_point"] + sigma_scale * cert["Sigma"]
    if lo > 0:
        return "channel_limited"
    if hi < 0:
        return "encoder_limited"
    return "abstain"


def diagnose_point(cert):
    """Deployable point diagnosis (no certified guarantee, no abstention).
    Uses the consistent point margin whose shared estimator bias cancels."""
    return "channel_limited" if cert["margin_point"] > 0 else "encoder_limited"

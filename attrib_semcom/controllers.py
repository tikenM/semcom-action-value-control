"""Controllers and cause-agnostic SOTA signals.

Action model: a per-decision budget admits ONE action.
  channel-directed (power): s -> s+1   (cleaner channel)
  encoder-directed        : e -> e+1   (richer/retrained encoder)
Each policy chooses an action; we measure realized eval error, oracle-action
agreement, budget, and (with the risk-control layer) coverage.
"""
import numpy as np
from . import decomposition as dec
from . import conformal as cf


def _after(backend, e, s, action, n, seed):
    if action == "power":
        return backend.evaluate(e, s + 1, n=n, seed=seed)
    if action == "encoder":
        return backend.evaluate(e + 1, s, n=n, seed=seed)
    return backend.evaluate(e, s, n=n, seed=seed)   # no-op


def oracle_action(backend, e, s, n, seed):
    ep = cf.error_rate(_after(backend, e, s, "power", n, seed))
    ee = cf.error_rate(_after(backend, e, s, "encoder", n, seed))
    return ("power", ep) if ep < ee else ("encoder", ee)


def stabilized_oracle_action(backend, e, s, n, seed, n_draws=5):
    """Oracle action selection averaged over multiple independent noise draws.

    A single-draw oracle_action can be unstable when the true error gap
    between the two actions is comparable to the per-draw noise floor (this
    is the mechanism behind STL-10's committed-point accuracy shortfall in
    the certified layer: Sec. IV-E/IV-K of the main paper). This variant
    draws n_draws independent evaluations per action, using deterministically
    offset seeds so results remain reproducible, and averages realized error
    before selecting. It is a drop-in replacement for oracle_action with an
    identical return signature: (action_name, mean_error_of_selected_action).

    Cost: n_draws x the evaluation cost of oracle_action. Use n_draws=1 to
    recover the original single-draw behavior exactly.
    """
    errs_power, errs_encoder = [], []
    for k in range(n_draws):
        draw_seed = seed + 10_000 * k   # deterministic, reproducible offset
        errs_power.append(cf.error_rate(_after(backend, e, s, "power", n, draw_seed)))
        errs_encoder.append(cf.error_rate(_after(backend, e, s, "encoder", n, draw_seed)))
    ep = float(np.mean(errs_power))
    ee = float(np.mean(errs_encoder))
    return ("power", ep) if ep < ee else ("encoder", ee)


# ------------------------------------------------- cause-agnostic SOTA signals
# All signals use ONLY current-operating-point information -- no trying candidate
# actions. This matches exactly what the decomposition diagnosis sees; giving a
# baseline action look-ahead would hand it the oracle's privilege (unavailable to
# any real controller, since 'try the encoder upgrade' means switching to another
# trained model and re-transmitting).

def signal_channel_snr(backend, e, s, gamma_thresh):
    """Physical-layer cause heuristic: low SNR -> blame channel (power),
    high SNR -> blame encoder. The natural current-point cause competitor."""
    return "power" if backend.gamma(s) < gamma_thresh else "encoder"


def signal_confidence_default(backend, e, s, default="power"):
    """Decoder confidence signals THAT the link fails, not WHY: both regimes
    raise predictive entropy, so as an allocation rule it can only fall back to a
    fixed default. This gap is the crux of the contribution."""
    return default


def signal_feature_importance(backend, e, s):
    """CIBA-style importance is defined on features and routes budget to protect
    transmission -> channel-directed, regardless of cause."""
    return "power"


def signal_ber_proxy(backend, e, s):
    """Hard-decision BER surrogate attributes failure to the channel -> power."""
    return "power"


# ---------------------------------------------------------- proposed policy
def diagnose_then_act(backend, e, s, n, seed, T, sigma_scale=1.0,
                      upper_mode="dpi_capacity", gate=None, mode="point"):
    """Diagnosis -> action class.
    mode='point'     : deployable point diagnosis (no abstention) drives action.
    mode='certified' : Theorem-2 diagnosis; abstain -> layer default action.
    """
    rec = backend.evaluate(e, s, n=n, seed=seed)
    cert = dec.certified_losses(rec, backend.K, T=T, upper_mode=upper_mode)
    if mode == "certified":
        diag = dec.diagnose_certified(cert, sigma_scale=sigma_scale)
    else:
        diag = dec.diagnose_point(cert)
    if diag == "encoder_limited":
        action = "encoder"
    elif diag == "channel_limited":
        action = "power"
    else:                                   # abstain -> layer default
        action = "power" if gate is None else gate
    return action, diag, cert


# ---------------------------------------------- Fix 1: calibrated selection
def calibrated_action(backend, e, s, n, seed, T, maps, upper_mode="dpi_capacity"):
    """Rank actions by the isotonic-CALIBRATED predicted error reduction rather
    than by raw loss dominance. Returns (action, v_power, v_encoder)."""
    rec = backend.evaluate(e, s, n=n, seed=seed)
    cert = dec.certified_losses(rec, backend.K, T=T, upper_mode=upper_mode)
    v_pow = maps["power"].predict(cert["Lch_point"])
    v_enc = maps["encoder"].predict(cert["Lenc_point"])
    return ("power" if v_pow >= v_enc else "encoder"), v_pow, v_enc


# --------------------------------------- Fix 2: SNR-prior fusion controller
def fused_snr_action(backend, e, s, n, seed, T, maps, gamma_thresh,
                     override_margin=0.0, upper_mode="dpi_capacity"):
    """Default to the channel-SNR heuristic; override with the calibrated
    diagnosis ONLY when the calibrated action-values disagree with the SNR pick
    by more than override_margin. Guarantees 'never worse than SNR except on a
    confident, calibrated disagreement'."""
    snr_pick = signal_channel_snr(backend, e, s, gamma_thresh)
    diag_pick, v_pow, v_enc = calibrated_action(
        backend, e, s, n, seed, T, maps, upper_mode)
    gap = abs(v_pow - v_enc)
    if diag_pick != snr_pick and gap > override_margin:
        return diag_pick, True
    return snr_pick, False


# ------------------- GP-surrogate baseline (re-implementation of [gpcr]) -----
# Faithful to the MECHANISM of constraint-Bayesian-optimization rate selection:
# learn a surrogate of task performance as a function of OBSERVABLE operating-point
# features (channel quality and compression setting), then select the action whose
# predicted post-action performance is best. The contrast with our controller is
# exactly the paper's thesis: this surrogate sees physical/rate features but NOT
# the encoder/channel loss decomposition, so it cannot resolve cause.

def fit_gp_surrogate(backend, e_vals, s_vals, alpha=1e-4):
    """Fit a GP mapping (log SNR, rate k) -> realized task error on the
    calibration split. Returns a predictor usable at unseen operating points."""
    import warnings
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    X, y = [], []
    for e in e_vals + [max(e_vals) + 1]:
        for s in s_vals + [max(s_vals) + 1]:
            try:
                rec = (backend.evaluate_cal(e, s)
                       if hasattr(backend, "evaluate_cal")
                       else backend.evaluate(e, s, n=6000, seed=90000 + 7 * e + s))
            except (KeyError, IndexError):
                continue
            err = float(np.mean(rec.post_noisy.argmax(1) != rec.y))
            X.append([np.log10(max(backend.gamma(s), 1e-6)), float(rec.k)])
            y.append(err)
    X, y = np.asarray(X), np.asarray(y)
    if len(X) < 4:
        raise ValueError(
            f"GP surrogate needs >=4 calibration points, got {len(X)}. Check "
            "that e_vals/s_vals stay within the trained rate points and SNR map.")
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    k = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5) \
        + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=True)
    # Bound-adjustment ConvergenceWarnings are expected and harmless; force-ignore
    # them locally so a project-wide warnings-as-errors setting cannot silently
    # sink this baseline (this was a suspected real-world failure mode).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit((X - Xm) / Xs, y)
    return dict(gp=gp, Xm=Xm, Xs=Xs)


def gp_surrogate_action(backend, e, s, n, seed, surrogate):
    """Select the action whose predicted post-action error is lower."""
    gp, Xm, Xs = surrogate["gp"], surrogate["Xm"], surrogate["Xs"]
    cand = {}
    for a in ("power", "encoder"):
        e2, s2 = (e, s + 1) if a == "power" else (e + 1, s)
        try:
            rec = backend.evaluate(e2, s2, n=n, seed=seed)
        except (KeyError, IndexError):
            continue
        feat = np.array([[np.log10(max(backend.gamma(s2), 1e-6)), float(rec.k)]])
        cand[a] = float(gp.predict((feat - Xm) / Xs)[0])
    if not cand:
        return "power"
    return min(cand, key=cand.get)

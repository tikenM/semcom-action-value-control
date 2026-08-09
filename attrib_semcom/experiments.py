"""Experiment orchestrator for the Q1 program.

Produces, over an operating-point grid:
  Q1 regime separation among equal-error points
  Q2 certified-diagnosis accuracy vs oracle-best action (+ abstention rate)
  Q3 policy value: realized error / oracle-action agreement / coverage for
     diagnose-then-act vs cause-agnostic baselines, with significance tests
  SOTA action-selection comparison across cause-agnostic signals
"""
import os
import numpy as np
from . import decomposition as dec
from . import conformal as cf
from . import controllers as ctl
from . import stats as st


def fit_global_temperature(backend, e_vals, s_vals, n=4000):
    """Pool a labelled calibration split across the grid; fit one temperature.
    Kept for the closed-form backend, where all rate points share one decoder
    family and pooling is appropriate. For real-data backends with sharply
    different rate points, prefer fit_per_e_temperature (see below)."""
    posts, ys = [], []
    has_cal = hasattr(backend, "evaluate_cal")
    for e in e_vals:
        for s in s_vals:
            if has_cal:
                rec = backend.evaluate_cal(e, s)
            else:
                rec = backend.evaluate(e, s, n=n // (len(e_vals) * len(s_vals)) + 1,
                                       seed=1000 + 7 * e + s)
            posts.append(rec.post_noisy); ys.append(rec.y)
    P = np.concatenate(posts); Y = np.concatenate(ys)
    return dec.fit_temperature(P, Y)


def fit_per_e_temperature(backend, e_vals, s_vals):
    """Fit ONE temperature PER encoder-quality index e, pooling only across
    channel states. Use this whenever rate points differ sharply (e.g. k=1 vs
    k=6): each deployed encoder has its own confidence profile, and a single
    pooled temperature systematically miscalibrates the extremes, which biases
    the point-margin estimate exactly where the two loss terms are noisiest.
    Returns {e: T}."""
    has_cal = hasattr(backend, "evaluate_cal")
    T_by_e = {}
    for e in e_vals:
        posts, ys = [], []
        for s in s_vals:
            rec = backend.evaluate_cal(e, s) if has_cal else backend.evaluate(
                e, s, n=1000, seed=1000 + 7 * e + s)
            posts.append(rec.post_noisy); ys.append(rec.y)
        P = np.concatenate(posts); Y = np.concatenate(ys)
        T_by_e[e] = dec.fit_temperature(P, Y)
    return T_by_e


def eval_seed(e, s, seed0=0):
    """The per-(e,s) evaluation seed used by run_program and every downstream
    audit. Any script that computes error rates at (e, s) must use THIS seed to
    stay paired with run_program's rows -- otherwise noise draws differ and the
    audit's helped/hurt classification measures noise variance, not action
    quality. Kept as a single-source-of-truth function so the convention can
    never drift between scripts."""
    return int(seed0) + 101 * int(e) + int(s)


def run_program(backend, e_vals, s_vals, n=6000, target=0.35, alpha=0.10,
                sigma_scale=1.0, upper_mode="dpi_capacity", T=None, seed0=0,
                oracle_n_draws=1):
    """oracle_n_draws=1 reproduces the exact prior single-draw oracle
    (unchanged results on every existing cached run). oracle_n_draws>1 uses
    controllers.stabilized_oracle_action, averaging that many independent
    noise draws per action before selecting -- recommended for datasets
    where the true action gap can be comparable to single-draw noise (see
    Sec. IV-E/IV-K of the main paper on STL-10's oracle instability)."""
    if T is None:
        T = fit_global_temperature(backend, e_vals, s_vals)
    per_e = isinstance(T, dict)

    # current-point channel-SNR threshold (median gamma over the grid)
    gamma_thresh = float(np.median([backend.gamma(s) for s in s_vals]))

    # Fix 1: fit isotonic action-value maps offline on the calibration split
    from . import calibration as cal
    maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T, upper_mode)
    # GP-surrogate baseline (re-implementation of the closest prior action-value
    # method); fit on the same calibration split as our isotonic maps
    gp_sur = None
    if os.environ.get("SKIP_GP") != "1":
        try:
            gp_sur = ctl.fit_gp_surrogate(backend, e_vals, s_vals)
        except Exception:
            import traceback, sys
            print("[experiments.run_program] GP-surrogate baseline failed to fit; "
                  "excluding it from this run. Full traceback:", file=sys.stderr)
            traceback.print_exc()
            gp_sur = None

    rows = []
    for e in e_vals:
        T_e = T[e] if per_e else T
        for s in s_vals:
            seed = eval_seed(e, s, seed0)
            truth = backend.true_losses(e, s)
            gt_regime = ("encoder_limited" if truth["L_enc"] > truth["L_ch"]
                         else "channel_limited")
            if oracle_n_draws > 1:
                o_act, o_err = ctl.stabilized_oracle_action(
                    backend, e, s, n, seed, n_draws=oracle_n_draws)
            else:
                o_act, o_err = ctl.oracle_action(backend, e, s, n, seed)

            # proposed deployable policy: point diagnose-then-act
            p_act, diag_pt, cert = ctl.diagnose_then_act(
                backend, e, s, n, seed, T_e, sigma_scale, upper_mode, mode="point")
            p_err = cf.error_rate(ctl._after(backend, e, s, p_act, n, seed))
            # certified diagnosis (guarantee layer) recorded separately
            diag_cert = dec.diagnose_certified(cert, sigma_scale=sigma_scale)

            # Fix 1: calibrated action-value selection
            cal_act, _, _ = ctl.calibrated_action(
                backend, e, s, n, seed, T_e, maps, upper_mode)
            cal_err = cf.error_rate(ctl._after(backend, e, s, cal_act, n, seed))
            # Fix 2: SNR-prior fusion (override SNR only on confident disagreement)
            fus_act, overrode = ctl.fused_snr_action(
                backend, e, s, n, seed, T_e, maps, gamma_thresh, 0.0, upper_mode)
            fus_err = cf.error_rate(ctl._after(backend, e, s, fus_act, n, seed))

            # baselines
            base_acts = {
                "power_only": "power",
                "encoder_only": "encoder",
                "confidence_default": ctl.signal_confidence_default(backend, e, s),
                "channel_snr": ctl.signal_channel_snr(backend, e, s, gamma_thresh),
                "feature_importance": ctl.signal_feature_importance(backend, e, s),
                "ber_proxy": ctl.signal_ber_proxy(backend, e, s),
            }
            if gp_sur is not None:
                base_acts["gp_surrogate"] = ctl.gp_surrogate_action(
                    backend, e, s, n, seed, gp_sur)
            base_err = {k: cf.error_rate(ctl._after(backend, e, s, a, n, seed))
                        for k, a in base_acts.items()}

            rows.append(dict(
                e=e, s=s, gt_regime=gt_regime,
                true_Lenc=truth["L_enc"], true_Lch=truth["L_ch"],
                true_margin=truth["L_ch"] - truth["L_enc"], base_err0=truth["bayes"],
                oracle_action=o_act, oracle_err=o_err,
                diag=diag_pt, diag_cert=diag_cert, diag_action=p_act, diag_err=p_err,
                cal_action=cal_act, cal_err=cal_err,
                fused_action=fus_act, fused_err=fus_err, fused_overrode=overrode,
                est_margin=cert["margin_point"], Sigma=cert["Sigma"],
                base_acts=base_acts, base_err=base_err))
    return dict(T=T, rows=rows, target=target, alpha=alpha,
                sigma_scale=sigma_scale, upper_mode=upper_mode)


# --------------------------------------------------------------- audits
def override_audit(backend, e_vals, s_vals, T, maps, n=6000, seed0=0,
                   upper_mode="dpi_capacity"):
    """Per-override helped/hurt/neutral classification, using the SAME per-(e,s)
    evaluation seed as run_program. Every override compares fused vs SNR under
    an IDENTICAL noise realization (paired at the seed level, not just at the
    action level), so the observed delta measures action quality rather than
    noise-draw variance. Returned dict is JSON-serializable and matches the
    format Sec. IV-G of the paper reads from.

    A single shared implementation for every dataset (vision and non-vision):
    calling backend.evaluate directly, without a seed, or evaluating twice per
    action would break the seed-level pairing and inflate the neutral count."""
    per_e = isinstance(T, dict)
    gamma_thresh = float(np.median([backend.gamma(s) for s in s_vals]))
    helped = hurt = neutral = 0
    deltas = []
    for e in e_vals:
        T_e = T[e] if per_e else T
        for s in s_vals:
            seed = eval_seed(e, s, seed0)
            fus_act, overrode = ctl.fused_snr_action(
                backend, e, s, n, seed, T_e, maps, gamma_thresh, 0.0, upper_mode)
            if not overrode:
                continue
            snr_act = ctl.signal_channel_snr(backend, e, s, gamma_thresh)
            err_f = cf.error_rate(ctl._after(backend, e, s, fus_act, n, seed))
            err_s = cf.error_rate(ctl._after(backend, e, s, snr_act, n, seed))
            d = err_s - err_f          # positive => override helped
            deltas.append(d)
            if   d >  1e-6: helped += 1
            elif d < -1e-6: hurt += 1
            else:           neutral += 1
    n_over = helped + hurt + neutral
    return dict(n_overrides=n_over, helped=helped, hurt=hurt, neutral=neutral,
                mean_delta=float(np.mean(deltas)) if deltas else 0.0,
                total_delta=float(np.sum(deltas)) if deltas else 0.0,
                harm_rate=float(hurt / n_over) if n_over else 0.0)


# --------------------------------------------------------------- analyses
def analyze(res):
    rows = res["rows"]
    out = {}

    # Q1 regime separation among near-equal task-error points
    B0 = np.array([r["base_err0"] for r in rows])
    enc_lim = np.array([r["true_Lenc"] > r["true_Lch"] for r in rows])
    edges = np.linspace(B0.min(), B0.max(), 9)
    binid = np.clip(np.digitize(B0, edges) - 1, 0, len(edges) - 2)
    mixed, total = 0, 0
    bin_report = []
    for b in range(len(edges) - 1):
        idx = np.where(binid == b)[0]
        if len(idx) < 4:
            continue
        frac = float(np.mean(enc_lim[idx])); total += 1
        mixed += (0.2 < frac < 0.8)
        bin_report.append((float(edges[b]), float(edges[b + 1]), len(idx), frac))
    out["Q1"] = dict(bins=bin_report, mixed=mixed, total=total,
                     holds=mixed >= 2)

    # Q2 diagnosis accuracy vs oracle
    def to_class(a): return "channel_limited" if a == "power" else "encoder_limited"
    # (a) deployable POINT diagnosis: broad, no abstention
    point_acc = np.mean([r["diag"] == to_class(r["oracle_action"]) for r in rows])
    pc = np.mean([r["oracle_action"] == "power" for r in rows])
    base_rate = max(pc, 1 - pc)
    # (b) CERTIFIED diagnosis: correctness on committed points + abstain rate
    committed = [r for r in rows if r["diag_cert"] != "abstain"]
    cert_frac = len(committed) / len(rows)
    cert_acc = (np.mean([r["diag_cert"] == to_class(r["oracle_action"])
                         for r in committed]) if committed else float("nan"))
    out["Q2"] = dict(accuracy=float(point_acc), majority_baseline=float(base_rate),
                     certified_accuracy=float(cert_acc) if committed else None,
                     certified_committed_fraction=float(cert_frac),
                     certified_abstain_fraction=float(1 - cert_frac))

    # Q3 policy value + significance
    diag_err = np.array([r["diag_err"] for r in rows])
    cal_err = np.array([r["cal_err"] for r in rows])
    fused_err = np.array([r["fused_err"] for r in rows])
    baselines = ["power_only", "encoder_only", "confidence_default", "channel_snr",
                 "feature_importance", "ber_proxy"]
    if rows and "gp_surrogate" in rows[0]["base_acts"]:
        baselines.append("gp_surrogate")
    oracle_err = np.array([r["oracle_err"] for r in rows])
    noact = np.array([r["base_err0"] for r in rows])
    pol = {}
    # the FUSED policy is the headline proposed method; significance is measured
    # against it so 'never worse than SNR' is directly testable
    ref = fused_err
    for name, arr in [("diagnose_then_act", diag_err),
                      ("calibrated", cal_err), ("fused_snr", fused_err)]:
        m, lo, hi = st.bootstrap_ci(arr)
        pol[name] = dict(mean=m, ci=(lo, hi))
    for b in baselines:
        be = np.array([r["base_err"][b] for r in rows])
        m, lo, hi = st.bootstrap_ci(be)
        _, p = st.wilcoxon_signed_rank(be, ref)        # baseline vs FUSED
        d, pb = st.paired_bootstrap_diff(be, ref)
        pol[b] = dict(mean=m, ci=(lo, hi), wilcoxon_p=p,
                      mean_diff_vs_prop=d, boot_p=pb)
    m, lo, hi = st.bootstrap_ci(oracle_err); pol["oracle"] = dict(mean=m, ci=(lo, hi))
    m, lo, hi = st.bootstrap_ci(noact); pol["no_action"] = dict(mean=m, ci=(lo, hi))
    gain = (noact.mean() - fused_err.mean()) / (noact.mean() - oracle_err.mean() + 1e-9)
    # fraction of points where fusion overrode the SNR prior
    override_frac = float(np.mean([r["fused_overrode"] for r in rows]))
    out["Q3"] = dict(policies=pol, oracle_gain_captured=float(gain),
                     fusion_override_fraction=override_frac)
    sig = np.array([r["Sigma"] for r in rows])
    out["Sigma"] = dict(mean=float(sig.mean()), std=float(sig.std(ddof=1)),
                        min=float(sig.min()), q25=float(np.percentile(sig,25)),
                        median=float(np.median(sig)),
                        q75=float(np.percentile(sig,75)), max=float(sig.max()))

    # SOTA action-selection agreement with oracle
    sota = {}
    def acc_of_actions(actions):
        return float(np.mean([to_class(a) == to_class(r["oracle_action"])
                              for a, r in zip(actions, rows)]))
    for b in baselines:
        sota[b] = acc_of_actions([r["base_acts"][b] for r in rows])
    sota["diagnose_then_act(point)"] = out["Q2"]["accuracy"]
    sota["calibrated"] = acc_of_actions([r["cal_action"] for r in rows])
    sota["fused_snr"] = acc_of_actions([r["fused_action"] for r in rows])
    out["SOTA"] = sota

    # coverage of the guarantee layer under the (fused) proposed policy
    cov = cf.coverage_report(fused_err, res["target"])
    lo, hi = st.clopper_pearson(int(round(cov["coverage"] * len(rows))), len(rows))
    out["coverage"] = dict(**cov, ci=(lo, hi), target=res["target"])
    return out

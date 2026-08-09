"""Tighter I_tx upper bound: CLUB applied to CLEAN-path posteriors.

Uses the same contrastive log-ratio upper bound (Cheng et al. 2020) already
used by the paper for the received side, applied at a different location: the
noiseless-channel decoder posteriors. This bounds I(Z_tx; Y) non-trivially and
is a candidate replacement for the trivial alphabet bound I_tx_hi = H(Y).

The mathematical validity rests on the same calibration assumption the paper
already makes for the received-side CLUB: the (temperature-calibrated) decoder
posteriors approximate the true conditional p(Y | Z_tx).

WARNING: club_upper() inflates for confident decoders. On CLEAN-path posteriors
from a well-trained vision decoder, the decoder is highly confident and CLUB
may exceed the trivial H(Y). The code clips to min(H(Y), CLUB), so a bad CLUB
just gives no improvement. Whether it works on real vision systems is
therefore an empirical question this phase answers.

Reports for each backend:
    Sigma_trivial      mean Sigma with I_tx_hi = H(Y)
    Sigma_club_clean   mean Sigma with I_tx_hi = min(H(Y), CLUB(post_clean))
    commits_half_*     half-width certified commits under each rule
    commits_full_*     full-width certified commits under each rule (Theorem 3
                       operational status)
    acc_on_commits     accuracy on the committed points, per rule and band
    tightening_bits    mean(Sigma_trivial - Sigma_club_clean); negative means
                       CLUB does not help on this system.
"""
import json
import numpy as np
from attrib_semcom import experiments as ex
from attrib_semcom.decomposition import (apply_temperature, barber_agakov_lower,
                                         club_upper, latent_capacity, HY)


def _true_class(backend, e, s):
    tl = backend.true_losses(e, s)
    return "encoder" if tl["L_enc"] > tl["L_ch"] else "channel"


def _itx_upper_club_clean(post_clean, K, itx_lo):
    """CLUB on clean path, clipped so it never exceeds H(Y) or falls below
    the lower estimate."""
    return max(min(HY(K), club_upper(post_clean, K)), itx_lo)


def evaluate_backend(backend, e_vals, s_vals, label, per_e_T=True):
    """Run BOTH rules (trivial H(Y) upper, CLUB-clean upper) on the same grid
    with the same temperature; return diagnostics."""
    T = (ex.fit_per_e_temperature(backend, e_vals, s_vals) if per_e_T
         else ex.fit_global_temperature(backend, e_vals, s_vals))
    per_e = isinstance(T, dict)
    K = backend.K
    H = HY(K)

    out = {}
    for mode in ("trivial", "club_clean"):
        sigmas = []
        commits_full = commits_half = 0
        ok_full = ok_half = 0
        for e in e_vals:
            for s in s_vals:
                Te = T[e] if per_e else T
                rec = (backend.evaluate_cal(e, s)
                       if hasattr(backend, "evaluate_cal")
                       else backend.evaluate(e, s, n=6000, seed=9000 + 7*e + s))
                pc = apply_temperature(rec.post_clean, Te)
                pn = apply_temperature(rec.post_noisy, Te)
                itx_lo, _ = barber_agakov_lower(pc, rec.y, K); itx_lo = max(itx_lo, 0.0)
                irx_lo, _ = barber_agakov_lower(pn, rec.y, K); irx_lo = max(irx_lo, 0.0)
                irx_hi = max(min(H, latent_capacity(rec.k, rec.gamma)), irx_lo)
                if mode == "trivial":
                    itx_hi = H
                else:
                    itx_hi = _itx_upper_club_clean(pc, K, itx_lo)

                # margin interval and slack (paper Eq. 4-5)
                # Lenc range: [H - itx_hi, H - itx_lo]
                # Lch  range: [itx_lo - irx_hi, itx_hi - irx_lo]
                lenc_lo, lenc_hi = H - itx_hi, H - itx_lo
                lch_lo, lch_hi = max(itx_lo - irx_hi, 0.0), itx_hi - irx_lo
                lenc_lo = max(lenc_lo, 0.0)
                sigma = 0.5 * ((lch_hi - lenc_lo) - (lch_lo - lenc_hi))
                sigmas.append(float(sigma))

                margin = 2.0*itx_lo - irx_lo - H         # signed point margin
                truth = _true_class(backend, e, s)
                decl = "channel" if margin > 0 else "encoder"

                if abs(margin) > sigma:
                    commits_full += 1
                    ok_full += int(decl == truth)
                if abs(margin) > 0.5 * sigma:
                    commits_half += 1
                    ok_half += int(decl == truth)

        n = len(e_vals) * len(s_vals)
        out[mode] = dict(
            mean_sigma=float(np.mean(sigmas)),
            median_sigma=float(np.median(sigmas)),
            min_sigma=float(np.min(sigmas)),
            commits_full=int(commits_full),
            commits_half=int(commits_half),
            n=int(n),
            acc_full=(float(ok_full / commits_full) if commits_full else None),
            acc_half=(float(ok_half / commits_half) if commits_half else None),
        )

    trivial = out["trivial"]["mean_sigma"]
    club = out["club_clean"]["mean_sigma"]
    out["tightening_bits"] = float(trivial - club)
    out["helps"] = bool(club < trivial - 1e-6)
    return dict(label=label, **out)


def run_all_backends(backends, seeded_config, out_path="tighter_itx_results.json"):
    """`backends` is a dict {label: (backend, e_vals, s_vals, per_e_T)}."""
    results = {}
    for label, (b, e_vals, s_vals, per_e_T) in backends.items():
        print(f"[tighter_itx] evaluating {label} ...")
        results[label] = evaluate_backend(b, e_vals, s_vals, label, per_e_T)
        t = results[label]["tightening_bits"]
        h = results[label]["helps"]
        print(f"  -> mean Sigma: trivial={results[label]['trivial']['mean_sigma']:.3f} "
              f"club={results[label]['club_clean']['mean_sigma']:.3f}  "
              f"tightening={t:+.3f} bits  helps={h}")
        print(f"  -> full-width commits: "
              f"trivial={results[label]['trivial']['commits_full']}/"
              f"{results[label]['trivial']['n']}  "
              f"club={results[label]['club_clean']['commits_full']}/"
              f"{results[label]['club_clean']['n']}")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    return results

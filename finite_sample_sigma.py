"""Finite-sample uncertainty on the estimator slack Sigma, via bootstrap.

Theorems 1 and 3 of the main paper are population-valid: they hold
conditional on the one-sided information estimates being valid bounds on the
POPULATION mutual-information quantities, but Sigma as computed by
decomposition.certified_losses does not itself account for the SAMPLING
variability of those estimates from a finite calibration split (see the
Scope remark under Theorem 3 in the supplementary proofs).

This script quantifies that finite-sample variability directly: at each
operating point, it resamples the calibration split with replacement B
times, refits the temperature and recomputes Sigma on each resample, and
reports a bootstrap percentile confidence interval on Sigma. This is the
same bootstrap methodology (percentile CI) the main paper already uses
elsewhere (Sec. IV-A, B=3000 for realized-error comparisons), applied here
to the estimator slack itself.

Two outputs:
  1. Per-point bootstrap CI on Sigma -- shows where the population-valid
     Sigma reported in the paper is a reliable point estimate vs. where
     finite-sample noise could shift it substantially.
  2. A pooled, single number: the largest relative CI width across the grid,
     as a single "how much should Sigma move due to finite-sample effects"
     headline statistic for the paper's Sec. III-E / Sec. IV-E discussion.

Usage:
    python finite_sample_sigma.py --dataset cifar10 --B 1000
    python finite_sample_sigma.py                      # controlled model, fast
"""
import argparse
import json
import numpy as np
from dataclasses import replace as dc_replace


def bootstrap_resample_rec(rec, rng):
    """Resample a Record's per-sample arrays with replacement, preserving
    the (post_clean, post_noisy, y) pairing. Scalar fields (k, gamma) are
    unchanged -- they are properties of the operating point, not the sample."""
    n = len(rec.y)
    idx = rng.integers(0, n, size=n)
    return dc_replace(rec, y=rec.y[idx], post_clean=rec.post_clean[idx],
                      post_noisy=rec.post_noisy[idx])


def bootstrap_sigma_at_point(backend, e, s, K, B, rng, upper_mode="dpi_capacity"):
    """Return an array of B bootstrap Sigma values at operating point (e,s).
    Refits temperature on EACH bootstrap resample, matching the offline
    protocol exactly (temperature is itself fit from the calibration split,
    so a faithful bootstrap must re-estimate it per resample)."""
    from attrib_semcom import decomposition as dec

    has_cal = hasattr(backend, "evaluate_cal")
    rec0 = backend.evaluate_cal(e, s) if has_cal else backend.evaluate(
        e, s, n=1000, seed=1000 + 7 * e + s)

    sigmas = np.empty(B)
    for b in range(B):
        rec_b = bootstrap_resample_rec(rec0, rng)
        T_b = dec.fit_temperature(rec_b.post_noisy, rec_b.y)
        cert_b = dec.certified_losses(rec_b, K, T=T_b, upper_mode=upper_mode)
        sigmas[b] = cert_b["Sigma"]
    return sigmas


def run(backend, e_vals, s_vals, K, B=1000, seed=0, upper_mode="dpi_capacity"):
    rng = np.random.default_rng(seed)
    rows = []
    for e in e_vals:
        for s in s_vals:
            sigmas = bootstrap_sigma_at_point(backend, e, s, K, B, rng, upper_mode)
            lo, hi = np.percentile(sigmas, [2.5, 97.5])
            point_est = sigmas.mean()  # bootstrap mean; population Sigma is close to this
            rel_width = (hi - lo) / point_est if point_est > 1e-9 else float("nan")
            rows.append(dict(e=int(e), s=int(s),
                             sigma_mean=float(point_est),
                             sigma_ci_lo=float(lo), sigma_ci_hi=float(hi),
                             rel_ci_width=float(rel_width)))
            print(f"  (e={e},s={s})  Sigma={point_est:.4f}  "
                  f"95% CI=[{lo:.4f},{hi:.4f}]  rel_width={rel_width:.3f}")

    rel_widths = np.array([r["rel_ci_width"] for r in rows])
    max_rel_width = float(np.nanmax(rel_widths))
    mean_rel_width = float(np.nanmean(rel_widths))
    print()
    print(f"Max relative CI width across grid:  {max_rel_width:.3f}")
    print(f"Mean relative CI width across grid: {mean_rel_width:.3f}")
    print()
    print("Interpretation: this is the fraction by which finite-sample noise")
    print("could plausibly move Sigma at the worst (or average) operating")
    print("point, at 95% bootstrap confidence, relative to the population-valid")
    print("Sigma reported elsewhere in the paper.")

    return dict(rows=rows, max_rel_ci_width=max_rel_width,
                mean_rel_ci_width=mean_rel_width, B=B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="dataset NAME for a trained vision backend "
                         "(mnist, cifar10, stl10); omit for the closed-form model")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--B", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upper-mode", default="dpi_capacity", choices=["dpi_capacity", "club"])
    args = ap.parse_args()

    if args.dataset is None:
        from attrib_semcom.backends import ControlledBackend
        backend = ControlledBackend(chan_decay=0.80)
        e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
        K = backend.K
        label = "controlled"
    else:
        from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG,
                                         DATASET_SPECS, pick_device)
        cfg = SUGGESTED_CONFIG[args.dataset]
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
            data_root=args.data_root, width=cfg["width"], epochs=cfg["epochs"],
            device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=0)
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))
        K = DATASET_SPECS[args.dataset]["K"]
        label = args.dataset

    print(f"BOOTSTRAP SIGMA  ({label}, B={args.B} replicates per operating point)")
    print("=" * 66)
    out = run(backend, e_vals, s_vals, K, B=args.B, seed=args.seed,
             upper_mode=args.upper_mode)
    out["dataset"] = label
    fn = f"bootstrap_sigma_{label}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()

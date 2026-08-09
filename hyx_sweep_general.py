"""H(Y|X) sensitivity sweep, generalized to any backend.

Table II of the main paper sweeps a hypothetical residual entropy H(Y|X) on
the CONTROLLED model only, because that is the one setting with ground truth
to validate against. This script runs the SAME sweep methodology (subtract a
hypothetical H(Y|X) from the estimated Lenc, recompute declarations and
half-width commits) on trained vision backends, where no ground truth on
declarations exists but the SENSITIVITY of the diagnosis to Assumption 1's
violation can still be reported directly -- exactly what the requested
revision asks for.

Two complementary uses:

  1. HYPOTHETICAL SWEEP (works everywhere, no new assumptions):
       python hyx_sweep_general.py --dataset cifar10
     Sweeps H(Y|X) in {0.00, 0.05, 0.10, 0.20, 0.30, 0.50} bits and reports,
     per value, how many declarations flip relative to the H(Y|X)=0 baseline
     and how many points the half-width certified rule still commits on.

  2. RESIDUAL-ENTROPY ESTIMATE via Fano inversion (--reference-error):
     If a defensible reference/near-ceiling error rate Pe_ref is available
     for the dataset (e.g. a well-documented near-human or near-ceiling
     classifier error rate from the literature -- NOT invented here; supply
     your own citable number), Fano's inequality gives an approximate UPPER
     bound on the irreducible H(Y|X) consistent with that error rate:
         H(Y|X) <~ H_b(Pe_ref) + Pe_ref * log2(K-1)
     This is the same Fano bound the main paper already uses in the other
     direction (Theorem 1); here it is inverted to turn a credible error
     floor into a credible entropy ceiling. Passing --reference-error runs
     the sweep only up to that estimated ceiling, rather than an arbitrary
     range, and reports the single implied H(Y|X) value explicitly.
     THE REFERENCE ERROR RATE MUST BE SUPPLIED BY THE USER WITH A CITATION;
     this script does not assume one, since an uncited number would be worse
     than an explicit hypothetical sweep.

Usage:
    python hyx_sweep_general.py --dataset cifar10
    python hyx_sweep_general.py --dataset stl10 --reference-error 0.015
    python hyx_sweep_general.py                      # controlled model
"""
import argparse
import json
import numpy as np


def hy_of_K(K):
    return float(np.log2(K))


def fano_invert_residual_entropy(Pe_ref, K):
    """Approximate upper bound on H(Y|X) implied by a reference error rate,
    via Fano's inequality: H(Y|X) <= H_b(Pe) + Pe*log2(K-1)."""
    Pe = max(min(Pe_ref, 1 - 1.0 / K), 1e-9)
    Hb = -(Pe * np.log2(Pe) + (1 - Pe) * np.log2(1 - Pe)) if 0 < Pe < 1 else 0.0
    return float(Hb + Pe * np.log2(max(K - 1, 1)))


def run_sweep(backend, e_vals, s_vals, K, T, hyx_values, sigma_scale_committed=0.5):
    from attrib_semcom import decomposition as dec

    per_e = isinstance(T, dict)
    has_cal = hasattr(backend, "evaluate_cal")

    baseline_decls, point_margins, sigmas = {}, {}, {}
    for e in e_vals:
        T_e = T[e] if per_e else T
        for s in s_vals:
            rec = (backend.evaluate_cal(e, s) if has_cal
                   else backend.evaluate(e, s, n=6000, seed=1000 + 7*e + s))
            cert = dec.certified_losses(rec, K, T=T_e)
            m0 = cert["margin_point"]  # FIX: was cert["Lch_point"] - cert["Lenc_point"],
            # which uses Lch_point = max(Itx_lo - Irx_lo, 0.0) -- clipped at
            # zero -- and diverges from decomposition.py's own purpose-built
            # margin_point (unclipped, "the shared estimator bias cancels in
            # the difference") whenever Itx_lo < Irx_lo. experiments.py and
            # decomposition.diagnose_point both already use margin_point;
            # this brings hyx_sweep_general.py into line with them.
            baseline_decls[(e, s)] = "channel" if m0 > 0 else "encoder"
            point_margins[(e, s)] = m0
            sigmas[(e, s)] = cert["Sigma"]

    n_total = len(baseline_decls)
    rows = []
    for hyx in hyx_values:
        flips = commits_half = 0
        for k in baseline_decls:
            # Sec III-A: assuming H(Y|X)=0 when the truth is h overestimates
            # Lenc by h, so the signed margin (Lch - Lenc) is underestimated
            # by h; correcting for a hypothetical true H(Y|X)=hyx means
            # adding hyx back to the point margin.
            m_swept = point_margins[k] + hyx
            new_decl = "channel" if m_swept > 0 else "encoder"
            if new_decl != baseline_decls[k]:
                flips += 1
            if abs(m_swept) > sigma_scale_committed * sigmas[k]:
                commits_half += 1
        rows.append(dict(hyx=float(hyx), flips=int(flips), n=int(n_total),
                         committed_half=int(commits_half)))
    return dict(rows=rows, n=n_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="mnist, cifar10, or stl10; omit for the controlled model")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reference-error", type=float, default=None,
                    help="a CITED reference/near-ceiling error rate for this "
                         "dataset; if given, estimates a single H(Y|X) ceiling "
                         "via Fano inversion instead of an arbitrary sweep")
    args = ap.parse_args()

    if args.dataset is None:
        from attrib_semcom.backends import ControlledBackend
        from attrib_semcom import experiments as ex
        backend = ControlledBackend(chan_decay=0.80)
        e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
        K = backend.K
        T = ex.fit_global_temperature(backend, e_vals, s_vals)
        label = "controlled"
    else:
        from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG,
                                         DATASET_SPECS, pick_device)
        from attrib_semcom import experiments as ex
        cfg = SUGGESTED_CONFIG[args.dataset]
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
            data_root=args.data_root, width=cfg["width"], epochs=cfg["epochs"],
            device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=0)
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))
        K = DATASET_SPECS[args.dataset]["K"]
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        label = args.dataset

    if args.reference_error is not None:
        h_ceiling = fano_invert_residual_entropy(args.reference_error, K)
        print(f"Fano-inverted H(Y|X) ceiling from reference error "
              f"{args.reference_error}: {h_ceiling:.4f} bits")
        hyx_values = np.linspace(0, h_ceiling, 6)
    else:
        hyx_values = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50)

    print(f"H(Y|X) SWEEP  ({label})")
    print("=" * 60)
    out = run_sweep(backend, e_vals, s_vals, K, T, hyx_values)
    n = out["n"]
    for r in out["rows"]:
        print(f"  H(Y|X)={r['hyx']:.4f}  flips={r['flips']}/{n}  "
              f"committed(half)={r['committed_half']}/{n}")

    out["dataset"] = label
    out["reference_error"] = args.reference_error
    fn = f"hyx_sweep_{label}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()

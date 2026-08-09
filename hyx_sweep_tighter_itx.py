"""H(Y|X) sensitivity sweep under the TIGHTER (non-trivial) transmitted-side
upper bound Iˆ↑_tx of Eq. (4), extending hyx_sweep_general.py.

ADDRESSES REVIEWER MAJOR WEAKNESS #3 (Assumption-1 analysis incomplete for
the full pipeline):
    "The entire decomposition rests on this assumption. Sensitivity tables
    only count point-diagnosis flips under the trivial transmitted-side
    bound. They do not re-evaluate the full controller + non-trivial CLUB
    diagnostic under realistic residual entropy on the vision datasets."

hyx_sweep_general.py (behind the paper's Table II and Table III) sweeps a
hypothetical residual entropy H(Y|X) and reports declaration flips and
half-width commits computed under the TRIVIAL transmitted-side bound
I^up_tx = H(Y) only -- Table II/III's own captions say so explicitly. But
Sec. III-D's tighter, CLUB-based estimate Iˆ↑_tx is the version "recommended
for deployment" (Sec. III-F) and Table IV/tighter_itx.py already show it
substantially changes certified commit counts on CIFAR-10 and STL-10 (e.g.
STL-10: 0/28 commits under the trivial bound vs 28/28 under the tighter
bound, at the default full-width band). Nothing in the codebase currently
re-runs the H(Y|X) sensitivity sweep under the bound the paper actually
recommends. This script closes that gap.

Mechanics, and why only Sigma (not the declaration) needs to be recomputed:
  Sec. III-D's own text: "Substituting Iˆ↑_tx for H(Y) in the slack Sigma
  reduces the transmitted-side gap" -- only the SLACK term changes; the
  point margin M = 2*Itx_lo - Irx_lo - H(Y) (Eq. 5, evaluated at the lower
  transmitted-side estimate as tighter_itx.py already does) does not depend
  on the upper estimate Itx_hi at all. So this script computes the margin
  ONCE per operating point (shared with the trivial-bound sweep) and TWO
  slacks -- Sigma_trivial (Itx_hi = H(Y)) and Sigma_tighter (Itx_hi =
  min(H(Y), CLUB(post_clean)), matching tighter_itx.py's construction
  exactly) -- then sweeps H(Y|X) against both, reporting flips (which are
  IDENTICAL between trivial/tighter, since flips depend only on the margin)
  and commits (which DIFFER, since commits depend on Sigma) side by side.

Usage:
    python hyx_sweep_tighter_itx.py --dataset stl10
    python hyx_sweep_tighter_itx.py --dataset cifar10
    python hyx_sweep_tighter_itx.py --dataset mnist
    python hyx_sweep_tighter_itx.py                       # controlled model

Output: prints a combined table and writes
    hyx_sweep_tighter_<dataset>.json
"""
import argparse
import json
import numpy as np
from attrib_semcom import experiments as ex
from attrib_semcom.decomposition import (apply_temperature, barber_agakov_lower,
                                         club_upper, latent_capacity, HY)


def _itx_upper_club_clean(post_clean, K, itx_lo):
    """Identical construction to tighter_itx.py's helper of the same name,
    duplicated here (rather than imported) so this script has no import-time
    dependency on paper_outputs.tighter_itx beyond the shared decomposition
    primitives, matching this codebase's existing pattern of small,
    self-contained paper_outputs scripts."""
    return max(min(HY(K), club_upper(post_clean, K)), itx_lo)


def run_sweep(backend, e_vals, s_vals, K, T, hyx_values, sigma_scale_committed=0.5,
             per_e_T=True):
    """Sweep H(Y|X) under BOTH the trivial and tighter transmitted-side
    bounds. Returns dict with per-value flip/commit counts for each bound,
    computed from a SHARED margin (see module docstring)."""
    per_e = isinstance(T, dict) if per_e_T else False
    H = HY(K)

    baseline_decls = {}     # (e,s) -> "channel" | "encoder", at H(Y|X)=0
    point_margins = {}      # (e,s) -> signed point margin at H(Y|X)=0
    sigmas_trivial = {}     # (e,s) -> Sigma under trivial bound
    sigmas_tighter = {}     # (e,s) -> Sigma under tighter bound

    for e in e_vals:
        Te = T[e] if per_e else T
        for s in s_vals:
            rec = (backend.evaluate_cal(e, s)
                   if hasattr(backend, "evaluate_cal")
                   else backend.evaluate(e, s, n=6000, seed=1000 + 7 * e + s))
            pc = apply_temperature(rec.post_clean, Te)
            pn = apply_temperature(rec.post_noisy, Te)
            itx_lo, _ = barber_agakov_lower(pc, rec.y, K); itx_lo = max(itx_lo, 0.0)
            irx_lo, _ = barber_agakov_lower(pn, rec.y, K); irx_lo = max(irx_lo, 0.0)
            irx_hi = max(min(H, latent_capacity(rec.k, rec.gamma)), irx_lo)

            itx_hi_trivial = H
            itx_hi_tighter = _itx_upper_club_clean(pc, K, itx_lo)

            def sigma_of(itx_hi):
                lenc_lo = max(H - itx_hi, 0.0)
                lenc_hi = H - itx_lo
                lch_lo = max(itx_lo - irx_hi, 0.0)
                lch_hi = itx_hi - irx_lo
                return 0.5 * ((lch_hi - lenc_lo) - (lch_lo - lenc_hi))

            # point margin, shared between both bounds (Sec. III-D)
            margin0 = 2.0 * itx_lo - irx_lo - H

            baseline_decls[(e, s)] = "channel" if margin0 > 0 else "encoder"
            point_margins[(e, s)] = margin0
            sigmas_trivial[(e, s)] = sigma_of(itx_hi_trivial)
            sigmas_tighter[(e, s)] = sigma_of(itx_hi_tighter)

    n_total = len(baseline_decls)
    rows = []
    for hyx in hyx_values:
        flips = 0
        commits_half_trivial = commits_full_trivial = 0
        commits_half_tighter = commits_full_tighter = 0
        for k in baseline_decls:
            m_swept = point_margins[k] + hyx
            new_decl = "channel" if m_swept > 0 else "encoder"
            if new_decl != baseline_decls[k]:
                flips += 1
            am = abs(m_swept)
            if am > sigma_scale_committed * sigmas_trivial[k]:
                commits_half_trivial += 1
            if am > sigmas_trivial[k]:
                commits_full_trivial += 1
            if am > sigma_scale_committed * sigmas_tighter[k]:
                commits_half_tighter += 1
            if am > sigmas_tighter[k]:
                commits_full_tighter += 1
        rows.append(dict(
            hyx=float(hyx), flips=int(flips), n=int(n_total),
            committed_half_trivial=int(commits_half_trivial),
            committed_full_trivial=int(commits_full_trivial),
            committed_half_tighter=int(commits_half_tighter),
            committed_full_tighter=int(commits_full_tighter),
        ))
    return dict(rows=rows, n=n_total,
               mean_sigma_trivial=float(np.mean(list(sigmas_trivial.values()))),
               mean_sigma_tighter=float(np.mean(list(sigmas_tighter.values()))))


def print_report(out, label):
    n = out["n"]
    print(f"\n{'='*78}")
    print(f"H(Y|X) SWEEP UNDER TRIVIAL vs. TIGHTER TRANSMITTED-SIDE BOUND  --  {label}")
    print(f"{'='*78}")
    print(f"  n operating points: {n}")
    print(f"  mean Sigma: trivial={out['mean_sigma_trivial']:.3f} bits  "
          f"tighter={out['mean_sigma_tighter']:.3f} bits")
    print()
    hdr = (f"  {'H(Y|X)':>8s}  {'flips':>8s}  "
          f"{'commit(half) triv':>18s}  {'commit(full) triv':>18s}  "
          f"{'commit(half) tight':>19s}  {'commit(full) tight':>19s}")
    print(hdr)
    for r in out["rows"]:
        print(f"  {r['hyx']:8.2f}  {r['flips']:>4d}/{n:<3d}  "
              f"{r['committed_half_trivial']:>10d}/{n:<3d}     "
              f"{r['committed_full_trivial']:>10d}/{n:<3d}     "
              f"{r['committed_half_tighter']:>10d}/{n:<3d}      "
              f"{r['committed_full_tighter']:>10d}/{n:<3d}")
    print()
    r0 = out["rows"][0]
    if r0["committed_full_tighter"] > r0["committed_full_trivial"]:
        print(f"  NOTE: at H(Y|X)=0, the tighter bound commits on "
              f"{r0['committed_full_tighter']}/{n} points at full width versus "
              f"{r0['committed_full_trivial']}/{n} under the trivial bound. Every "
              f"one of those additional commits is a claim this script does NOT "
              f"itself validate against a violation of Assumption 1 -- it only "
              f"reports how many of them would FLIP declaration under a swept "
              f"H(Y|X), which is the sensitivity check this script adds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="mnist, cifar10, or stl10; omit for the controlled model")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--hyx-values", default="0.00,0.05,0.10,0.20,0.30,0.50",
                    help="comma-separated H(Y|X) values to sweep, in bits")
    args = ap.parse_args()
    hyx_values = tuple(float(x) for x in args.hyx_values.split(","))

    if args.dataset is None:
        from attrib_semcom.backends import ControlledBackend
        backend = ControlledBackend(chan_decay=0.80)
        e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
        K = backend.K
        T = ex.fit_global_temperature(backend, e_vals, s_vals)
        label = "controlled"
        per_e_T = False
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
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        label = args.dataset
        per_e_T = True

    print(f"[hyx_sweep_tighter_itx] running on {label} ...")
    out = run_sweep(backend, e_vals, s_vals, K, T, hyx_values, per_e_T=per_e_T)
    print_report(out, label)

    out["dataset"] = label
    out["hyx_values"] = list(hyx_values)
    fn = f"hyx_sweep_tighter_{label}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()

"""Minimal, faithful reconstruction of the pieces of experiments.py needed to
reproduce table2.py's run() exactly, using the ACTUAL decomposition.py and
backends.py source (uploaded this session), to compute the real,
verified Table II numbers under both the original (buggy) and fixed margin
convention -- not an estimate, an actual run of the real logic.

fit_global_temperature is reconstructed here exactly as it appeared earlier
in this session (from experiments.py, index 4 of the original document
set): pool a labelled calibration split across the grid, fit one
temperature via decomposition.fit_temperature.
"""
import numpy as np
import decomposition as dec
from backends import ControlledBackend


def fit_global_temperature(backend, e_vals, s_vals, n=4000):
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


def run(hyx_values=(0.00, 0.10, 0.20, 0.30, 0.50),
        e_vals=None, s_vals=None, sigma_scale_committed=0.5,
        use_fixed_margin=False):
    """Faithful reconstruction of table2.py's run(), with a flag to select
    the original (buggy) or fixed margin convention, so both can be computed
    from the identical evaluation batch for a direct, apples-to-apples
    comparison."""
    if e_vals is None: e_vals = list(range(0, 6))
    if s_vals is None: s_vals = list(range(0, 8))

    b = ControlledBackend(chan_decay=0.80)
    T = fit_global_temperature(b, e_vals, s_vals)

    baseline_decls = {}
    point_margins = {}
    sigmas = {}
    for e in e_vals:
        for s in s_vals:
            rec = b.evaluate(e, s, n=6000, seed=1000 + 7*e + s)
            cert = dec.certified_losses(rec, b.K, T=T)
            if use_fixed_margin:
                m0 = cert["margin_point"]
            else:
                m0 = cert["Lch_point"] - cert["Lenc_point"]   # the original code
            baseline_decls[(e, s)] = "channel" if m0 > 0 else "encoder"
            point_margins[(e, s)] = m0
            sigmas[(e, s)] = cert["Sigma"]

    n_total = len(baseline_decls)
    rows = []
    for hyx in hyx_values:
        flips = 0
        commits_half = 0
        for k in baseline_decls:
            m_swept = point_margins[k] + hyx
            new_decl = "channel" if m_swept > 0 else "encoder"
            if new_decl != baseline_decls[k]:
                flips += 1
            if abs(m_swept) > sigma_scale_committed * sigmas[k]:
                commits_half += 1
        rows.append(dict(hyx=float(hyx), flips=int(flips), n=int(n_total),
                         committed_half=int(commits_half)))

    return dict(sigma_scale_committed=float(sigma_scale_committed),
               grid=dict(e_vals=list(map(int, e_vals)),
                         s_vals=list(map(int, s_vals)), n=int(n_total)),
               rows=rows, mean_sigma=float(np.mean(list(sigmas.values()))))


if __name__ == "__main__":
    print("="*72)
    print("Margin convention A: Lch_point - Lenc_point")
    print("="*72)
    out_orig = run(use_fixed_margin=False)
    print(f"  mean Sigma = {out_orig['mean_sigma']:.4f}")
    for r in out_orig["rows"]:
        print(f"  H(Y|X)={r['hyx']:.2f}  flips={r['flips']}/{r['n']}  "
              f"committed(half)={r['committed_half']}/{r['n']}")

    print()
    print("="*72)
    print("Margin convention B: cert['margin_point']")
    print("="*72)
    out_fixed = run(use_fixed_margin=True)
    print(f"  mean Sigma = {out_fixed['mean_sigma']:.4f}")
    for r in out_fixed["rows"]:
        print(f"  H(Y|X)={r['hyx']:.2f}  flips={r['flips']}/{r['n']}  "
              f"committed(half)={r['committed_half']}/{r['n']}")

    print()
    print("="*72)
    print("DIRECT COMPARISON AT H(Y|X)=0")
    print("="*72)
    print(f"  Manuscript Table II value: 24/48")
    print(f"  Reconstruction (Lch_point - Lenc_point): {out_orig['rows'][0]['committed_half']}/48")
    print(f"  Reconstruction (margin_point):           {out_fixed['rows'][0]['committed_half']}/48")

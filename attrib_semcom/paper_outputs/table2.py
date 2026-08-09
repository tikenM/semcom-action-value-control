"""Table II production: hypothetical H(Y|X) sweep on the primary controlled model.

Sec. IV-E of the paper: sweep a hypothetical residual entropy H(Y|X), subtract
it from the estimated encoder loss, and compare the resulting declarations
against those at H(Y|X)=0. For each H(Y|X) value, report:
  - number of declarations that flip vs. the H(Y|X)=0 baseline
  - number of points on which the half-width certified rule commits

This is the HYPOTHETICAL sweep on the primary controlled model, distinct from
the ambiguous_source.py experiment (which uses genuine H(Y|X) via label noise).
"""
import json
import numpy as np
from attrib_semcom.backends import ControlledBackend
from attrib_semcom import experiments as ex, decomposition as dec


def run(hyx_values=(0.00, 0.10, 0.20, 0.30, 0.50),
        e_vals=None, s_vals=None, sigma_scale_committed=0.5):
    """Return the Table II rows plus baseline metadata.

    sigma_scale_committed=0.5 = the "half-width certified" rule the paper
    reports in Table II's third column.
    """
    if e_vals is None: e_vals = list(range(0, 6))
    if s_vals is None: s_vals = list(range(0, 8))

    b = ControlledBackend(chan_decay=0.80)
    T = ex.fit_global_temperature(b, e_vals, s_vals)

    # First pass: baseline declarations at H(Y|X)=0, and Sigma per point (which
    # is independent of H(Y|X): the sweep only shifts the POINT margin, not the
    # bound-interval WIDTHS).
    baseline_decls = {}   # (e,s) -> "encoder" | "channel"
    point_margins = {}    # (e,s) -> signed point margin at H(Y|X)=0
    sigmas = {}           # (e,s) -> Sigma
    for e in e_vals:
        for s in s_vals:
            rec = b.evaluate(e, s, n=6000, seed=1000 + 7*e + s)
            cert = dec.certified_losses(rec, b.K, T=T)
            # signed point margin at H(Y|X)=0 baseline: Lch - Lenc
            m0 = cert["margin_point"]  # FIX: was cert["Lch_point"] - cert["Lenc_point"],
            # which uses Lch_point = max(Itx_lo - Irx_lo, 0.0) -- clipped at
            # zero -- and diverges from decomposition.py's own purpose-built
            # margin_point (unclipped, "the shared estimator bias cancels in
            # the difference") whenever Itx_lo < Irx_lo. experiments.py and
            # decomposition.diagnose_point both already use margin_point;
            # this brings table2.py into line with them.
            baseline_decls[(e, s)] = "channel" if m0 > 0 else "encoder"
            point_margins[(e, s)] = m0
            sigmas[(e, s)] = cert["Sigma"]

    n_total = len(baseline_decls)
    rows = []
    for hyx in hyx_values:
        # Sec III-A: assuming H(Y|X)=0 when the truth is h inflates Lenc by h
        # (we use I(X;Y)=H(Y) but the truth is I(X;Y)=H(Y)-h, so Lenc = I(X;Y)-Itx
        # is over-estimated by h). Signed margin M = Lch - Lenc is therefore
        # UNDER-estimated by h. Correcting the estimate under hypothetical true
        # H(Y|X)=hyx means adding hyx back to the margin.
        flips = 0
        commits_half = 0
        for k in baseline_decls:
            m_swept = point_margins[k] + hyx
            new_decl = "channel" if m_swept > 0 else "encoder"
            if new_decl != baseline_decls[k]:
                flips += 1
            # half-width certified: commit iff |margin| > (1/2) * Sigma
            if abs(m_swept) > sigma_scale_committed * sigmas[k]:
                commits_half += 1
        rows.append(dict(hyx=float(hyx),
                         flips=int(flips), n=int(n_total),
                         committed_half=int(commits_half)))

    return dict(sigma_scale_committed=float(sigma_scale_committed),
                grid=dict(e_vals=list(map(int, e_vals)),
                          s_vals=list(map(int, s_vals)),
                          n=int(n_total)),
                rows=rows)


if __name__ == "__main__":
    out = run()
    print("H(Y|X) sweep on primary controlled model (Table II)")
    print(f"  grid: {out['grid']['n']} operating points; "
          f"half-width rule at sigma_scale={out['sigma_scale_committed']}")
    print(f"  {'H(Y|X)':>8s}  {'flips':>10s}  {'committed':>14s}")
    for r in out["rows"]:
        print(f"  {r['hyx']:8.2f}  {r['flips']:>4d}/{r['n']:<4d}   "
              f"{r['committed_half']:>4d}/{r['n']:<4d}")
    json.dump(out, open("table2_hyx_sweep.json", "w"), indent=2)
    print("wrote table2_hyx_sweep.json")

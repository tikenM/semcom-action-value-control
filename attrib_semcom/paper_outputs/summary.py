"""Human-readable summary of the whole reproduction, written to paper_summary.txt.

Reads every cached JSON in results/ and produces a one-page summary of headline
numbers so the user can quickly compare to the manuscript.
"""
import os, json


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _mean(d, key):
    return d.get("summary", {}).get(key, {}).get("mean")


def _std(d, key):
    return d.get("summary", {}).get(key, {}).get("std")


def write_summary(results_root, out_path):
    lines = []
    W = lines.append

    W("="*72)
    W("Paper 2 reproduction summary")
    W("="*72)
    W("")

    # SEEDED per dataset
    W("MULTI-SEED RESULTS")
    W("-"*72)
    for k in ["mnist", "cifar10", "stl10"]:
        d = _read_json(os.path.join(results_root, k, "seeded.json"))
        if d is None:
            W(f"  {k:8s}  [no seeded.json]"); continue
        seeds = d.get("seeds", "?")
        W(f"  {k:8s}  ({seeds} seeds)")
        for key in ["raw", "calibrated", "fused", "channel_snr", "gp_surrogate",
                    "err_fused", "err_snr", "err_gp",
                    "oracle_gain", "override_frac"]:
            m = _mean(d, key); s = _std(d, key)
            if m is None: continue
            W(f"    {key:15s} {m:.4f} +/- {(s or 0):.4f}")
    W("")

    # WINE
    w = _read_json(os.path.join(results_root, "wine", "seeded.json"))
    if w is None:
        w = _read_json(os.path.join(results_root, "wine", "wine_results_5seed.json"))
    if w:
        W("WINE NON-VISION PILOT")
        W("-"*72)
        for key in ["raw","calibrated","fused","channel_snr","gp_surrogate",
                    "err_fused","err_snr","err_gp","oracle_gain"]:
            m = _mean(w, key); s = _std(w, key)
            if m is None: continue
            W(f"    {key:15s} {m:.4f} +/- {(s or 0):.4f}")
        pa = w.get("pooled_audit")
        if pa:
            W(f"    pooled_override_audit: n={pa.get('n')} helped={pa.get('helped')} "
              f"hurt={pa.get('hurt')} harm_rate={pa.get('harm_rate'):.3f}")
        W("")

    # TIGHTER I_TX
    tit = _read_json(os.path.join(results_root, "tighter_itx_all.json"))
    if tit:
        W("TIGHTER I_tx (CLUB on clean-path posteriors)")
        W("-"*72)
        for label, r in tit.items():
            W(f"  {label:16s}  Sigma trivial={r['trivial']['mean_sigma']:.3f}  "
              f"club_clean={r['club_clean']['mean_sigma']:.3f}  "
              f"tightening={r['tightening_bits']:+.3f} bits  helps={r['helps']}")
            W(f"                    full-width commits: "
              f"trivial={r['trivial']['commits_full']}/{r['trivial']['n']}  "
              f"club_clean={r['club_clean']['commits_full']}/{r['club_clean']['n']}")
            if r['club_clean']['acc_full'] is not None:
                W(f"                    accuracy on club_clean commits (full): "
                  f"{r['club_clean']['acc_full']:.3f}")
        W("")

    # TABLE II
    t2 = _read_json(os.path.join(results_root, "controlled", "hyx_sweep.json"))
    if t2:
        W("H(Y|X) SENSITIVITY (Table II)")
        W("-"*72)
        n = t2["grid"]["n"]
        for r in t2["rows"]:
            W(f"    H(Y|X)={r['hyx']:.2f}  flips={r['flips']}/{n}  "
              f"half-width committed={r['committed_half']}/{n}")
        W("")

    W("Wrote every table LaTeX fragment to results/tables/ and every figure PDF")
    W("to results/figures/. Diff seeded.json entries against the paper's Table IV")
    W("as the primary reproducibility check.")
    W("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)

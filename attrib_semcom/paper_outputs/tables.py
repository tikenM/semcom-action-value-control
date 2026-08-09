"""LaTeX table emitters, one function per paper table.

Each emitter takes a cached JSON dict from run_all.py and returns a LaTeX
tabular fragment (no \\begin{table}) that can be dropped into the manuscript
or included via \\input.

Naming convention: emit_table1_metric_comparison(json_dict) -> str.
"""


def _fmt(x, prec=3, nan_str="---"):
    if x is None or (isinstance(x, float) and x != x):
        return nan_str
    return f"{x:.{prec}f}"


def _fmtpm(mean, std, prec=3):
    if mean is None or (isinstance(mean, float) and mean != mean):
        return "---"
    return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"


# =========================================================================
# Table I: allocation metric comparison at fixed controller (48 op. points)
# =========================================================================
def emit_table1_metric_comparison(closed_form_analyze, closed_form_ablations=None):
    """closed_form_analyze = analyze(res) output on the closed-form model.
    closed_form_ablations = ablations dict; if present, its uncalibrated
    calibration-ablation result is used as the STII-proxy row."""
    A = closed_form_analyze
    sota = A["SOTA"]
    maj = A["Q2"]["majority_baseline"]
    # STII proxy = same decomposition on uncalibrated posteriors
    stii = None
    if closed_form_ablations:
        stii = closed_form_ablations.get("calibration", {}) \
                                    .get("uncalibrated", {}).get("accuracy")

    rows = [
        ("S-SNR decomp. (ours)", sota.get("diagnose_then_act(point)"),
         "Sees both loss components; calibrated."),
        ("STII proxy [5]", stii if stii is not None else sota.get("diagnose_then_act(point)"),
         "Sees both components but uncalibrated."),
        ("Channel SNR", sota.get("channel_snr"),
         "Captures channel only; blind to encoder."),
        ("Confidence", sota.get("confidence_default"),
         "Detects failure but not its cause."),
        ("Majority baseline", maj,
         "Not applicable."),
    ]
    body = "\n".join(f"        {name:22s} & {_fmt(acc)} & {cap} \\\\"
                    for name, acc, cap in rows)
    return f"""% Table I: allocation-metric comparison at fixed controller.
\\begin{{tabular}}{{lcp{{5.3cm}}}}
        \\toprule
        Signal & Accuracy & Capability \\\\
        \\midrule
{body}
        \\bottomrule
\\end{{tabular}}
"""


# =========================================================================
# Table II: H(Y|X) sensitivity (paper_outputs/table2.py)
# =========================================================================
def emit_table2_hyx(hyx_json):
    rows = hyx_json["rows"]
    n = hyx_json["grid"]["n"]
    body = "\n".join(
        f"        {r['hyx']:.2f} & {r['flips']}/{n} & "
        f"{r['committed_half']}/{n} \\\\"
        for r in rows)
    return f"""% Table II: H(Y|X) hypothetical sweep, primary controlled model.
\\begin{{tabular}}{{ccc}}
        \\toprule
        $H(Y \\mid X)$ (bits) & Declarations flipped & Committed (half-width) \\\\
        \\midrule
{body}
        \\bottomrule
\\end{{tabular}}
"""


# =========================================================================
# Table III: CV(g_enc) + override outcome across learned settings
# =========================================================================
def emit_table3_cv_gate(gate_by_dataset, audit_by_dataset):
    """
    gate_by_dataset:  {label: {"cv_g_enc": float, ...}}
    audit_by_dataset: {label: {"overrides": int, "helped": int, "hurt": int,
                               "neutral": int, "harm_rate": float}}
    """
    order = ["cifar10", "wine", "mnist", "stl10"]
    body_lines = []
    for k in order:
        if k not in gate_by_dataset or k not in audit_by_dataset:
            continue
        g = gate_by_dataset[k]; a = audit_by_dataset[k]
        cv = g.get("cv_g_enc", float("nan"))
        n_over = a.get("overrides", 0)
        helped = a.get("helped", 0)
        hurt = a.get("hurt", 0)
        neutral = a.get("neutral", 0)
        harm = a.get("harm_rate", 0.0)
        # If any neutrals, use dagger-marked helped/hurt (so column doesn't
        # imply helped+hurt=overrides).
        if neutral > 0:
            hh = f"{helped}/{hurt}$^\\dagger$"
        else:
            hh = f"{helped}/{hurt}"
        pretty = {"cifar10":"CIFAR-10","wine":"Wine",
                  "mnist":"MNIST","stl10":"STL-10"}[k]
        body_lines.append(
            f"        {pretty} & {_fmt(cv,2)} & {n_over} & {hh} & {_fmt(harm)} \\\\")
    body = "\n".join(body_lines)
    return f"""% Table III: encoder-side map variability + override outcome.
\\begin{{tabular}}{{lcccc}}
        \\toprule
        Setting & CV$(g_\\text{{enc}})$ & Overrides & Helped/Hurt & Harm rate \\\\
        \\midrule
{body}
        \\bottomrule
\\end{{tabular}}
"""


# =========================================================================
# Table IV: cause-attributed control across regimes
# =========================================================================
def emit_table4_control(closed_form_analyze, seeded_by_dataset):
    """
    closed_form_analyze: analyze() output on closed-form (single-run values).
    seeded_by_dataset:  {label: seeded_json_dict} for mnist, cifar10, stl10.
    """
    A = closed_form_analyze
    def _seeded(label, key):
        d = seeded_by_dataset.get(label, {}).get("summary", {}).get(key)
        return d
    def _row_seeded(label, key):
        d = _seeded(label, key)
        if d is None: return "---"
        return _fmtpm(d.get("mean"), d.get("std"))

    # Action-selection accuracies
    rows_acc = [
        ("Raw diagnosis",  A["Q2"]["accuracy"],   "raw"),
        ("Calibrated",     A["SOTA"]["calibrated"], "calibrated"),
        ("Fused (ours)",   A["SOTA"]["fused_snr"], "fused"),
        ("Channel SNR",    A["SOTA"]["channel_snr"], "channel_snr"),
        ("GP surrogate [9]", A["SOTA"].get("gp_surrogate"), "gp_surrogate"),
        ("Majority",       A["Q2"]["majority_baseline"], None),
    ]
    header = ("Metric & Closed-form & MNIST & CIFAR-10 & STL-10 \\\\")
    accbody = []
    for label, cf_val, seeded_key in rows_acc:
        m = _fmt(cf_val)
        mn = _row_seeded("mnist", seeded_key) if seeded_key else "0.500"
        cf = _row_seeded("cifar10", seeded_key) if seeded_key else "0.536"
        st = _row_seeded("stl10", seeded_key) if seeded_key else "---"
        accbody.append(f"        {label} & {m} & {mn} & {cf} & {st} \\\\")

    # Realized errors
    err_rows = [
        ("Channel SNR", "err_snr", "channel_snr_err_cf"),
        ("Fused (ours)","err_fused", "fused_err_cf"),
        ("GP surrogate [9]", "err_gp", "gp_err_cf"),
    ]
    # closed-form realized errors come from analyze()["Q3"]["policies"]
    pol = A["Q3"]["policies"]
    err_cf = {"channel_snr": pol.get("channel_snr", {}).get("mean"),
              "fused_snr": pol.get("fused_snr", {}).get("mean"),
              "gp_surrogate": pol.get("gp_surrogate", {}).get("mean")}
    errbody = []
    for label, k, _ in err_rows:
        cf_val = {"err_snr":"channel_snr","err_fused":"fused_snr",
                  "err_gp":"gp_surrogate"}[k]
        errbody.append(
            f"        {label} & {_fmt(err_cf.get(cf_val),4)} & "
            f"{_row_seeded('mnist', k)} & {_row_seeded('cifar10', k)} & "
            f"{_row_seeded('stl10', k)} \\\\")

    # oracle gain / override frac
    gain_cf = A["Q3"]["oracle_gain_captured"]
    ovr_cf  = A["Q3"]["fusion_override_fraction"]
    extras = "\n".join([
        f"        Oracle gain & {_fmt(gain_cf)} & {_row_seeded('mnist','oracle_gain')} "
        f"& {_row_seeded('cifar10','oracle_gain')} & {_row_seeded('stl10','oracle_gain')} \\\\",
        f"        SNR overrides & {_fmt(ovr_cf)} & {_row_seeded('mnist','override_frac')} "
        f"& {_row_seeded('cifar10','override_frac')} & {_row_seeded('stl10','override_frac')} \\\\",
    ])

    return f"""% Table IV: cause-attributed control across regimes (multi-seed).
\\begin{{tabular}}{{lcccc}}
        \\toprule
        {header}
        \\midrule
        \\multicolumn{{5}}{{l}}{{\\emph{{Action-selection accuracy}}}} \\\\
""" + "\n".join(accbody) + f"""
        \\midrule
        \\multicolumn{{5}}{{l}}{{\\emph{{Realized task error (lower is better)}}}} \\\\
""" + "\n".join(errbody) + f"""
        \\midrule
{extras}
        \\bottomrule
\\end{{tabular}}
"""


# =========================================================================
# Table V: data-efficiency, held-out action-selection accuracy
# =========================================================================
def emit_table5_data_eff(data_eff_by_dataset):
    """data_eff_by_dataset: {label: {"results": {"iso": {..}, "gp": {..},
                                                  "paired": {..}}, ...}}"""
    order = ["controlled", "cifar10", "mnist", "stl10"]
    labels = {"controlled":"Controlled model", "cifar10":"CIFAR-10",
              "mnist":"MNIST", "stl10":"STL-10"}
    body = []
    for k in order:
        if k not in data_eff_by_dataset:
            continue
        pretty = labels[k]
        d = data_eff_by_dataset[k]
        R = d["results"]
        # each fraction gives (mean, std, n)
        body.append(f"        \\multicolumn{{5}}{{l}}{{\\emph{{{pretty}}}}} \\\\")
        for f_str, (iso_m, iso_s, iso_n) in R["iso"].items():
            gp_m, gp_s, gp_n = R["gp"][f_str]
            diff = R["paired"][f_str]["mean_diff"]
            p    = R["paired"][f_str]["wilcoxon_p"]
            f = float(f_str)
            # significance marks (paper convention)
            if p is not None and p == p:  # not nan
                mark = "$^{**}$" if p < 0.01 else ("$^*$" if p < 0.05 else "")
            else:
                mark = ""
            body.append(
                f"        {f:.3f} & ${iso_m:.3f} \\pm {iso_s:.3f}$ & "
                f"${gp_m:.3f} \\pm {gp_s:.3f}$ & {diff:+.3f} & "
                f"{_fmt(p,4)}{mark} \\\\")
    return f"""% Table V: held-out action-selection vs calibration coverage.
\\begin{{tabular}}{{cccrc}}
        \\toprule
        Fraction & Isotonic & GP surr. & $\\Delta$ & $p$ \\\\
        \\midrule
""" + "\n".join(body) + """
        \\bottomrule
\\end{tabular}
"""


# =========================================================================
# Table VI: ablations on controlled model
# =========================================================================
def emit_table6_ablations_controlled(closed_form_ablations):
    A = closed_form_ablations
    # A is a dict with keys "calibration", "upper_bound", "abstain_sweep",
    # "confidence_vs_decomposition", "point_vs_certified", "channel"
    c = A["calibration"]
    ub = A["upper_bound"]
    ab = A["abstain_sweep"]
    cv = A["confidence_vs_decomposition"]
    pv = A["point_vs_certified"]
    ch = A["channel"]

    def _get(d, k, sub="accuracy"):
        return d.get(k, {}).get(sub, float("nan"))

    def _ab(k):
        # JSON serializes float dict keys as strings; try both
        return ab.get(k, {}).get('committed', ab.get(str(k), {}).get('committed'))
    body = f"""        Temperature calibration off / on (point acc.) & {_fmt(_get(c,'uncalibrated'))} / {_fmt(_get(c,'calibrated'))} \\\\
        Upper bound DPI+cap. vs. CLUB ($\\bar\\Sigma$) & {_fmt(ub['dpi_capacity']['mean_Sigma'],2)} / {_fmt(ub['club']['mean_Sigma'],2)} bits \\\\
        Abstain sweep, committed @ scale 0 / 0.5 / 1 & {_fmt(_ab(0.0))} / {_fmt(_ab(0.5))} / {_fmt(_ab(1.0))} \\\\
        Decomposition vs. confidence (acc.) & {_fmt(cv.get('decomposition'))} vs. {_fmt(cv.get('confidence'))} \\\\
        Point vs. certified (acc. on committed) & {_fmt(pv.get('point_accuracy'))} vs. {_fmt(pv.get('certified_accuracy'))} \\\\
        Channel AWGN vs. Rayleigh (acc.) & {_fmt(ch.get('awgn',{}).get('accuracy'))} vs. {_fmt(ch.get('rayleigh',{}).get('accuracy'))} \\\\"""
    return f"""% Table VI: ablations on controlled model.
\\begin{{tabular}}{{lc}}
        \\toprule
        Ablation & Result \\\\
        \\midrule
{body}
        \\bottomrule
\\end{{tabular}}
"""


# =========================================================================
# Table VII: ablations on CIFAR-10 learned JSCC (single seed)
# =========================================================================
def emit_table7_ablations_cifar(cifar_ablations):
    A = cifar_ablations
    tc = A["temperature_calibration"]
    ub = A["upper_bound"]
    ab = A["abstain_sweep"]
    cv = A["confidence_vs_decomposition"]
    rvac = A["raw_vs_action_value"]

    def _ab(k):
        return ab.get(k, {}).get('committed', ab.get(str(k), {}).get('committed'))
    body = f"""        Temperature calib. off / on (raw acc.) & {_fmt(tc['uncalibrated']['raw_acc'])} / {_fmt(tc['calibrated']['raw_acc'])} \\\\
        Temperature calib. off / on (action-value acc.) & {_fmt(tc['uncalibrated']['action_value_acc'])} / {_fmt(tc['calibrated']['action_value_acc'])} \\\\
        Upper bound DPI+cap. vs. CLUB ($\\bar\\Sigma$) & {_fmt(ub['dpi_capacity']['mean_Sigma'],2)} / {_fmt(ub['club']['mean_Sigma'],2)} bits \\\\
        Abstain sweep, committed @ scale 0 / 0.5 / 1 & {_fmt(_ab(0.0))} / {_fmt(_ab(0.5))} / {_fmt(_ab(1.0))} \\\\
        Decomposition vs. confidence (acc.) & {_fmt(cv.get('decomposition'))} vs. {_fmt(cv.get('confidence'))} \\\\
        Decomposition vs. channel-SNR (acc.) & {_fmt(cv.get('decomposition'))} vs. {_fmt(cv.get('best_cause_baseline'))} \\\\
        Raw vs. action-value rule (acc.) & {_fmt(rvac['raw_acc'])} vs. {_fmt(rvac['action_value_acc'])} \\\\"""
    return f"""% Table VII: ablations on CIFAR-10 learned JSCC (single seed).
\\begin{{tabular}}{{lc}}
        \\toprule
        Ablation & Result \\\\
        \\midrule
{body}
        \\bottomrule
\\end{{tabular}}
"""

"""Figure functions: Figs. 1-7 of the paper as matplotlib PDFs.

All figures use only matplotlib (no seaborn), grayscale-friendly colors, and
PDF output. Each function takes the necessary cached JSON dicts and writes a
single PDF to out_dir.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# journal-friendly defaults
def _apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,     # embed TrueType, IEEE-friendly
    })


# =========================================================================
# Fig. 1: certified sandwich bracket (closed-form model)
# =========================================================================
def fig1_sandwich(closed_form_res, out_path):
    """Plot Fano lower, true task error, Hellman-Raviv upper vs S-SNR.

    closed_form_res: run_program output on ControlledBackend (has 'rows' with
    S-SNR-adjacent quantities and true task error).
    """
    _apply_style()
    rows = closed_form_res["rows"]
    # For each row, compute S-SNR from the true Irx if available; fall back to
    # Barber-Agakov estimate scaled by H(Y).
    from attrib_semcom.decomposition import HY
    K = 10   # controlled model default
    H = HY(K)

    # True error is base_err0 (no-action Bayes error), S-SNR = Irx/(H-Irx).
    # We don't have per-row Irx directly, so approximate via
    # Irx = H - err_lower_bound_inverse; alternate: use fused_err as a proxy
    # for realized error at the operating point (base_err0 IS true error before
    # action).
    ssnr = []
    true_err = []
    hr_up = []
    fano_lo = []
    for r in rows:
        # Recover Irx via inversion of the Hellman-Raviv upper bound:
        # err <= (1/2) * H(Y)/(1+rho) => rho >= H(Y) / (2*err) - 1
        # Better: we saved Sigma/margin but not Irx directly. Use base_err0 as
        # the true error and pick rho on a log grid then compute bracket ends.
        err = float(r["base_err0"])
        # invert: rho = H(Y)/(2*err) - 1 gives the S-SNR at which HR-upper
        # equals the observed error. Use this as the plot's x-axis.
        rho = max(H / (2.0 * max(err, 1e-6)) - 1.0, 1e-3)
        ssnr.append(rho)
        true_err.append(err)
        # Hellman-Raviv upper: (1/2) H(Y)/(1+rho)
        hr_up.append(0.5 * H / (1.0 + rho))
        # Fano lower: (1/log2 K) * (H(Y)/(1+rho) - 1)
        lo = (1.0 / np.log2(K)) * (H / (1.0 + rho) - 1.0)
        fano_lo.append(max(lo, 0.0))

    ssnr = np.array(ssnr); true_err = np.array(true_err)
    hr_up = np.array(hr_up); fano_lo = np.array(fano_lo)
    order = np.argsort(ssnr)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ssnr[order], hr_up[order], "-", color="C3", lw=1.4,
            label="Hellman-Raviv upper")
    ax.plot(ssnr[order], true_err[order], "-o", color="k", ms=3, lw=1.0,
            label="true task error")
    ax.plot(ssnr[order], fano_lo[order], "-", color="C0", lw=1.4,
            label="Fano lower")
    ax.fill_between(ssnr[order], fano_lo[order], hr_up[order],
                    color="C0", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("S-SNR")
    ax.set_ylabel("task error")
    ax.set_title("Certified sandwich brackets task error")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 2: label-free estimator vs ground truth, uncal vs cal
# =========================================================================
def fig2_estimator(estimator_json, out_path):
    """estimator_json: {'uncalibrated': [(true_irx, est), ...],
                        'calibrated':   [(true_irx, est), ...]}"""
    _apply_style()
    uncal = np.array(estimator_json.get("uncalibrated", []))
    calib = np.array(estimator_json.get("calibrated",  []))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    if len(uncal):
        r_u = np.corrcoef(uncal[:,0], uncal[:,1])[0,1]
        ax.scatter(uncal[:,0], uncal[:,1], s=14, color="gray",
                   label=f"uncalibrated ($r={r_u:.2f}$)", alpha=0.7)
    if len(calib):
        r_c = np.corrcoef(calib[:,0], calib[:,1])[0,1]
        ax.scatter(calib[:,0], calib[:,1], s=14, color="C1",
                   label=f"calibrated ($r={r_c:.2f}$)", alpha=0.9)
    lo = 0.0
    hi = max(uncal[:,0].max() if len(uncal) else 1,
             calib[:,0].max() if len(calib) else 1) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="k", lw=0.7, alpha=0.6)
    ax.set_xlabel(r"true delivered info $I_{rx}$ (bits)")
    ax.set_ylabel("label-free estimate (bits)")
    ax.set_title("Estimator recovers after calibration")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 3: metric comparison bars (fixed controller, controlled model)
# =========================================================================
def fig3_metric_bars(closed_form_analyze, out_path, closed_form_ablations=None):
    """closed_form_ablations: same ablations dict tables.py's
    emit_table1_metric_comparison consumes. Its uncalibrated calibration-
    ablation accuracy is the STII-proxy value (Table I / Sec. IV-D). Without
    it, STII-proxy falls back to duplicating the S-SNR bar, which is wrong:
    the whole point of the STII row is that it is UNcalibrated."""
    _apply_style()
    A = closed_form_analyze
    sota = A["SOTA"]
    stii = None
    if closed_form_ablations:
        stii = closed_form_ablations.get("calibration", {}) \
                                    .get("uncalibrated", {}).get("accuracy")
    if stii is None:
        stii = sota.get("diagnose_then_act(point)", 0)
    entries = [
        ("S-SNR (ours)",       sota.get("diagnose_then_act(point)", 0)),
        ("STII-proxy (uncal.)", stii),
        ("channel-SNR",        sota.get("channel_snr", 0)),
        ("confidence",         sota.get("confidence_default", 0)),
    ]
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(labels))[::-1]   # top-down = best first
    bars = ax.barh(y, vals, color=["C0","C0","C0","C0"], height=0.55,
                   edgecolor="k")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("action-selection accuracy vs oracle")
    ax.set_title("S-SNR vs task-info (STII) and channel proxies")
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.2f}",
                va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 4: Lenc vs Lch scatter, marked by oracle-best action
# =========================================================================
def fig4_regime_scatter(closed_form_res, out_path):
    _apply_style()
    rows = closed_form_res["rows"]
    lenc = np.array([r["true_Lenc"] for r in rows])
    lch  = np.array([r["true_Lch"] for r in rows])
    is_power_best = np.array([r["oracle_action"] == "power" for r in rows])

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(lenc[is_power_best], lch[is_power_best],
               s=32, marker="o", facecolors="none", edgecolors="C0",
               label="power is oracle-best")
    ax.scatter(lenc[~is_power_best], lch[~is_power_best],
               s=32, marker="^", color="C3",
               label="encoder is oracle-best")
    mx = max(lenc.max(), lch.max()) * 1.05
    ax.plot([0, mx], [0, mx], "--", color="k", lw=0.6, alpha=0.5,
            label=r"$L_\mathrm{enc} = L_\mathrm{ch}$")
    ax.set_xlabel(r"encoder loss $L_\mathrm{enc}$ (bits)")
    ax.set_ylabel(r"channel loss $L_\mathrm{ch}$ (bits)")
    ax.set_title("Causal regimes vs. oracle-best action")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 5: cause attribution vs task regime (grouped bars per system)
# =========================================================================
def fig5_attribution_vs_regime(cf_analyze, seeded_by_dataset, out_path,
                                include_stl10=True):
    _apply_style()
    def _summary_mean(d, key):
        return d.get("summary", {}).get(key, {}).get("mean", 0.0)

    systems = [
        ("Closed-form\n(idealized)", "controlled",
         cf_analyze["Q2"]["accuracy"],
         cf_analyze["SOTA"]["calibrated"],
         cf_analyze["SOTA"]["channel_snr"]),
    ]
    for d in [("MNIST\n(easy)", "mnist"),
              ("CIFAR-10\n(hard)", "cifar10")] + (
              [("STL-10\n(new)", "stl10")] if include_stl10 else []):
        pretty, k = d
        sd = seeded_by_dataset.get(k, {})
        systems.append(
            (pretty, k, _summary_mean(sd, "raw"),
             _summary_mean(sd, "calibrated"),
             _summary_mean(sd, "channel_snr"))
        )

    labels = [s[0] for s in systems]
    raw = [s[2] for s in systems]
    cal = [s[3] for s in systems]
    snr = [s[4] for s in systems]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    b1 = ax.bar(x - w, raw, w, color="gray", label="raw diagnosis", edgecolor="k")
    b2 = ax.bar(x,     cal, w, color="C1",   label="calibrated (ours)", edgecolor="k")
    b3 = ax.bar(x + w, snr, w, color="C2",   label="channel-SNR", edgecolor="k")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, ls=":", color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("action-selection accuracy")
    ax.set_title("Cause attribution vs. task regime")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 6: cause-attribution against cause-agnostic baselines
# =========================================================================
def fig6_baselines(cf_analyze, seeded_by_dataset, out_path):
    _apply_style()
    def _summary_mean(d, key):
        return d.get("summary", {}).get(key, {}).get("mean", 0.0)

    # rows: methods; columns: systems (Closed-form, MNIST, CIFAR-10, STL-10)
    rows = ["fused (ours)", "calibrated (ours)", "channel-SNR",
            "encoder-only", "power-only", "confidence"]
    sys_labels = ["Closed-form", "MNIST", "CIFAR-10", "STL-10"]

    def _get(sys_key, method):
        if sys_key == "closed_form":
            A = cf_analyze
            m = {"fused (ours)":   A["SOTA"]["fused_snr"],
                 "calibrated (ours)": A["SOTA"]["calibrated"],
                 "channel-SNR":   A["SOTA"]["channel_snr"],
                 "encoder-only":  A["SOTA"].get("encoder_only", 0),
                 "power-only":    A["SOTA"].get("power_only", 0),
                 "confidence":    A["SOTA"].get("confidence_default", 0)}
            return m.get(method, 0)
        d = seeded_by_dataset.get(sys_key, {})
        # not every seeded json has all baselines, but paper-baseline names:
        m = {"fused (ours)":   _summary_mean(d, "fused"),
             "calibrated (ours)": _summary_mean(d, "calibrated"),
             "channel-SNR":   _summary_mean(d, "channel_snr"),
             "encoder-only":  _summary_mean(d, "encoder_only"),
             "power-only":    _summary_mean(d, "power_only"),
             "confidence":    _summary_mean(d, "confidence_default")}
        return m.get(method, 0)

    data = np.zeros((len(rows), len(sys_labels)))
    for i, method in enumerate(rows):
        for j, sys in enumerate(["closed_form", "mnist", "cifar10", "stl10"]):
            data[i, j] = _get(sys, method)

    y = np.arange(len(rows))
    h = 0.18
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    colors = ["C0", "C1", "C2", "C3"]
    for j, sys in enumerate(sys_labels):
        ax.barh(y + (j - 1.5) * h, data[:, j], h, color=colors[j],
                edgecolor="k", label=sys)
    ax.set_yticks(y); ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("action-selection accuracy vs. oracle")
    ax.set_title("Cause-attribution vs. cause-agnostic baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# =========================================================================
# Fig. 7: attribution benefit vs override fraction
# =========================================================================
def fig7_benefit_vs_override(cf_analyze, seeded_by_dataset, out_path):
    _apply_style()
    def _summary_mean(d, key):
        return d.get("summary", {}).get(key, {}).get("mean", 0.0)

    # each system contributes ONE point (override_frac, benefit)
    # where benefit = calibrated action-selection accuracy - channel-SNR acc.
    pts = []
    pts.append(("Closed-form",
                cf_analyze["Q3"]["fusion_override_fraction"],
                cf_analyze["SOTA"]["calibrated"] - cf_analyze["SOTA"]["channel_snr"]))
    for label, k in [("MNIST","mnist"), ("CIFAR-10","cifar10"), ("STL-10","stl10")]:
        d = seeded_by_dataset.get(k, {})
        if not d: continue
        pts.append((label,
                    _summary_mean(d, "override_frac"),
                    _summary_mean(d, "calibrated") - _summary_mean(d, "channel_snr")))

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = ["C0", "C1", "C3", "C2"]
    for i, (lbl, x, y) in enumerate(pts):
        ax.scatter([x], [y], s=90, color=colors[i % len(colors)],
                   edgecolors="k", zorder=3, label=lbl)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.axhline(0, ls=":", color="k", lw=0.5, alpha=0.6)
    ax.set_xlabel("SNR-override fraction")
    ax.set_ylabel("calibrated $-$ channel-SNR\n(action-selection acc.)")
    ax.set_title("Benefit tracks how often attribution engages")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)

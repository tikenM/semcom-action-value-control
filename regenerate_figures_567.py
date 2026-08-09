"""Standalone regenerator for Figs. 5, 6, 7 of the rewritten paper.

Reads cached JSONs (put them at the paths below, or edit the constants), and
writes fig5, fig6, fig7 as PDFs to the current directory. Uses only matplotlib.

Usage:
    python regenerate_figures_567.py
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Where to find the cached JSONs. Edit if your layout differs.
PATHS = dict(
    closed_form_analyze = "results/controlled/closed_form_analyze.json",
    mnist_seeded        = "results/mnist/seeded.json",
    cifar10_seeded      = "results/cifar10/seeded.json",
    stl10_seeded        = "results/stl10/seeded.json",
)

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def _rc():
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def _read(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} missing"); return None
    with open(path) as f: return json.load(f)


def _summary(d, key):
    v = d.get("summary", {}).get(key, {})
    return v.get("mean", 0.0), v.get("std", 0.0)


def fig5_attribution_vs_regime(cf, seeded, out):
    _rc()
    systems = [("Closed-form\n(idealized)",
                cf["Q2"]["accuracy"], cf["SOTA"]["calibrated"], cf["SOTA"]["channel_snr"])]
    for label, key in [("MNIST\n(easy)", "mnist"),
                       ("CIFAR-10\n(hard)", "cifar10"),
                       ("STL-10\n(hard, inverted raw)", "stl10")]:
        d = seeded.get(key)
        if d is None: continue
        raw_m, _   = _summary(d, "raw")
        cal_m, _   = _summary(d, "calibrated")
        snr_m, _   = _summary(d, "channel_snr")
        systems.append((label, raw_m, cal_m, snr_m))
    labels = [s[0] for s in systems]
    raw    = [s[1] for s in systems]
    cal    = [s[2] for s in systems]
    snr    = [s[3] for s in systems]
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    b1 = ax.bar(x - w, raw, w, color="gray", label="raw diagnosis", edgecolor="k")
    b2 = ax.bar(x,     cal, w, color="C1",   label="calibrated (ours)", edgecolor="k")
    b3 = ax.bar(x + w, snr, w, color="C2",   label="channel-SNR",  edgecolor="k")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, ls=":", color="k", lw=0.5, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("action-selection accuracy")
    ax.set_title("Cause attribution vs. task regime")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig6_baselines(cf, seeded, out):
    _rc()
    rows = ["fused (ours)", "calibrated (ours)", "channel-SNR",
            "encoder-only", "power-only", "confidence"]
    sys_keys   = ["closed_form", "mnist", "cifar10", "stl10"]
    sys_labels = ["Closed-form", "MNIST", "CIFAR-10", "STL-10"]

    def _val(sys, method):
        if sys == "closed_form":
            m = {"fused (ours)": cf["SOTA"]["fused_snr"],
                 "calibrated (ours)": cf["SOTA"]["calibrated"],
                 "channel-SNR": cf["SOTA"]["channel_snr"],
                 "encoder-only": cf["SOTA"].get("encoder_only", 0),
                 "power-only":   cf["SOTA"].get("power_only", 0),
                 "confidence":   cf["SOTA"].get("confidence_default", 0)}
            return m.get(method, 0)
        d = seeded.get(sys)
        if d is None: return 0
        m = {"fused (ours)":       _summary(d, "fused")[0],
             "calibrated (ours)":  _summary(d, "calibrated")[0],
             "channel-SNR":        _summary(d, "channel_snr")[0],
             "encoder-only":       _summary(d, "encoder_only")[0],
             "power-only":         _summary(d, "power_only")[0],
             "confidence":         _summary(d, "confidence_default")[0]}
        return m.get(method, 0)

    data = np.zeros((len(rows), len(sys_labels)))
    for i, method in enumerate(rows):
        for j, sys in enumerate(sys_keys):
            data[i, j] = _val(sys, method)

    y = np.arange(len(rows)); h = 0.19
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["C0", "C1", "C3", "C2"]
    for j, sys in enumerate(sys_labels):
        ax.barh(y + (j - 1.5) * h, data[:, j], h, color=colors[j],
                edgecolor="k", label=sys)
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("action-selection accuracy vs. oracle")
    ax.set_title("Cause-attribution vs. cause-agnostic baselines")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig7_benefit_vs_override(cf, seeded, out):
    _rc()
    pts = [("Closed-form",
            cf["Q3"]["fusion_override_fraction"],
            cf["SOTA"]["calibrated"] - cf["SOTA"]["channel_snr"])]
    for label, key in [("MNIST","mnist"), ("CIFAR-10","cifar10"), ("STL-10","stl10")]:
        d = seeded.get(key)
        if d is None: continue
        ovr = _summary(d, "override_frac")[0]
        ben = _summary(d, "calibrated")[0] - _summary(d, "channel_snr")[0]
        pts.append((label, ovr, ben))

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = ["C0", "C1", "C3", "C2"]
    for i, (lbl, x, yv) in enumerate(pts):
        ax.scatter([x], [yv], s=90, color=colors[i % len(colors)],
                   edgecolors="k", zorder=3)
        ax.annotate(lbl, (x, yv), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.axhline(0, ls=":", color="k", lw=0.5, alpha=0.6)
    ax.set_xlabel("SNR-override fraction")
    ax.set_ylabel("calibrated $-$ channel-SNR\n(action-selection acc.)")
    ax.set_title("Benefit tracks how often attribution engages")
    ax.grid(True, alpha=0.3)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def main():
    cf = _read(PATHS["closed_form_analyze"])
    seeded = {
        "mnist":   _read(PATHS["mnist_seeded"]),
        "cifar10": _read(PATHS["cifar10_seeded"]),
        "stl10":   _read(PATHS["stl10_seeded"]),
    }
    seeded = {k: v for k, v in seeded.items() if v is not None}
    if cf is None:
        raise SystemExit("closed_form_analyze.json is required; edit PATHS")
    fig5_attribution_vs_regime(cf, seeded, os.path.join(OUT_DIR, "fig_three_settings.pdf"))
    fig6_baselines            (cf, seeded, os.path.join(OUT_DIR, "fig_sota.pdf"))
    fig7_benefit_vs_override  (cf, seeded, os.path.join(OUT_DIR, "fig_value_vs_override.pdf"))


if __name__ == "__main__":
    main()

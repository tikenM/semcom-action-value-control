"""End-to-end reproducer for the Paper 2 experimental program.

Single command, all datasets, all tables, all figures. Delegates to existing
entry points where they exist (run_seeds.py, run_wine_seeds.py,
data_efficiency_sweep.py, get_gate_diagnostics.py, gate_sensitivity_sweep.py,
ambiguous_source.py) and calls new phases in attrib_semcom.paper_outputs for
outputs the paper needs but no shipped script produces (Table II sweep,
CLUB-on-clean-path evaluation on vision, Table VII driver, all figure PDFs,
all LaTeX table fragments).

USAGE
-----
Reproduce everything with default seeds (20 for MNIST/CIFAR/Wine, 10 for STL10):
    python run_all.py --all

Just re-emit figures & tables from cached JSON (no retraining):
    python run_all.py --phases figures,tables

Run a specific dataset through the seeded phase only:
    python run_all.py --datasets cifar10 --phases seeded

Reproduce the closed-form-only path (no torch needed, ~seconds):
    python run_all.py --phases closed_form,figures,tables

CACHE LAYOUT
------------
results/
    mnist/         seeded.json  data_eff.json  gate.json  tighter_itx.json
    cifar10/       seeded.json  data_eff.json  gate.json  ablations.json  tighter_itx.json
    stl10/         seeded.json  data_eff.json  gate.json  ablations.json  tighter_itx.json
    wine/          seeded.json  gate.json  tighter_itx.json
    controlled/    closed_form.json  hyx_sweep.json  ambiguous_source.json  ablations.json
    tighter_itx_all.json                  (aggregated across datasets)
    figures/       fig1.pdf ... fig7.pdf
    tables/        table1.tex ... table7.tex
    paper_summary.txt

PHASES
------
    closed_form  Run the closed-form controlled model once; produces the
                 basis for Figs 1, 4 and Tables I, VI, plus the H(Y|X) sweep
                 for Table II. Fast (seconds), no torch.
    seeded       Multi-seed replication of MNIST, CIFAR-10, STL-10 (vision)
                 and Wine (non-vision). Delegates to run_seeds.py and
                 run_wine_seeds.py. Requires torch + datasets.
    data_eff     Data-efficiency sweep (Table V) per dataset. Delegates to
                 data_efficiency_sweep.py.
    gate         CV(g_enc), gate decision, sensitivity sweep per dataset.
                 Delegates to get_gate_diagnostics.py and
                 gate_sensitivity_sweep.py.
    ablations    Table VII driver on trained vision backends (single seed).
    hyx          Table II hypothetical H(Y|X) sweep + ambiguous-source
                 validation on the closed-form model.
    tighter_itx  CLUB-on-clean-path evaluation on every backend, to decide
                 whether the tighter I_tx bound helps in practice.
    figures      Regenerate all figure PDFs from cached JSON.
    tables       Regenerate all LaTeX table fragments from cached JSON.
    summary      Write paper_summary.txt.

NEW DATASETS (paper doesn't currently contain STL-10)
-----------------------------------------------------
STL-10 is included as a fourth setting. Its results are novel: they do not
appear in the current paper. Compare against Table IV / Table V / Table III
to see how the framework generalizes to a higher-resolution vision task.
"""
import argparse
import json
import os
import subprocess
import sys
import time


ALL_DATASETS = ("mnist", "cifar10", "stl10", "wine")
VISION_DATASETS = ("mnist", "cifar10", "stl10")
ALL_PHASES = ("closed_form", "seeded", "data_eff", "gate", "ablations",
              "hyx", "tighter_itx", "figures", "tables", "summary")


# --------------------------------------------------------------- utilities
def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f: json.dump(obj, f, indent=2)


def read_json(path, default=None):
    if not os.path.exists(path): return default
    with open(path) as f: return json.load(f)


def phase_banner(name):
    print("\n" + "=" * 72)
    print(f"PHASE: {name}")
    print("=" * 72)


def _run_subprocess(cmd, cwd=None, log_path=None):
    """Run a subprocess, streaming its stdout, and optionally tee to log."""
    print(f"  $ {' '.join(cmd)}")
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "wb") as logf:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            for line in p.stdout:
                sys.stdout.buffer.write(line); sys.stdout.flush()
                logf.write(line)
            p.wait()
    else:
        p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise SystemExit(f"  [FAIL] {' '.join(cmd)} exit {p.returncode}")


# --------------------------------------------------------------- phases
def phase_closed_form(results_root, args):
    """Run the closed-form controlled model once, cache results + analyze()."""
    phase_banner("closed_form")
    from attrib_semcom.backends import ControlledBackend
    from attrib_semcom import experiments as ex
    from attrib_semcom import ablations as ab

    e_vals = list(range(0, 6)); s_vals = list(range(0, 8))
    b = ControlledBackend(chan_decay=0.80)
    res = ex.run_program(b, e_vals, s_vals, n=args.n, target=args.target,
                         alpha=args.alpha)
    A = ex.analyze(res)

    # write both the raw res (needed for Fig. 1 and Fig. 4) and the analyze()
    # output (needed for tables and Figs. 3, 5, 6, 7).
    write_json(os.path.join(results_root, "controlled", "closed_form_res.json"),
               _shrink_res(res))
    write_json(os.path.join(results_root, "controlled", "closed_form_analyze.json"), A)

    # Table VI: ablations on the controlled model
    print("  running controlled-model ablations for Table VI ...")
    abl = {
        "calibration": ab.ablation_calibration(b, e_vals, s_vals, n=args.n,
                                               target=args.target, alpha=args.alpha),
        "upper_bound": ab.ablation_upper_bound(b, e_vals, s_vals, n=args.n,
                                               target=args.target, alpha=args.alpha),
        "abstain_sweep": ab.ablation_abstain_sweep(b, e_vals, s_vals, n=args.n,
                                                   target=args.target, alpha=args.alpha),
        "confidence_vs_decomposition":
            ab.ablation_confidence_vs_decomposition(b, e_vals, s_vals, n=args.n,
                                                    target=args.target, alpha=args.alpha),
        "point_vs_certified":
            ab.ablation_point_vs_certified(b, e_vals, s_vals, n=args.n,
                                           target=args.target, alpha=args.alpha),
        "channel": ab.ablation_channel(
            lambda ch="awgn": ControlledBackend(
                chan_decay=(0.80 if ch=="awgn" else 0.88)),
            e_vals, s_vals, n=args.n, target=args.target, alpha=args.alpha),
    }
    write_json(os.path.join(results_root, "controlled", "ablations.json"), abl)
    print("  done.")


def _shrink_res(res):
    """Discard heavy arrays inside rows to keep the on-disk JSON small."""
    rows = []
    for r in res["rows"]:
        rr = {k: v for k, v in r.items()
              if k not in ("base_acts",)  # keep base_err, drop redundant
              }
        rows.append(rr)
    return dict(T=(res["T"] if not isinstance(res["T"], dict)
                   else {int(k): float(v) for k, v in res["T"].items()}),
                rows=rows, target=res["target"], alpha=res["alpha"],
                sigma_scale=res["sigma_scale"], upper_mode=res["upper_mode"])


def phase_seeded(results_root, args):
    """Dispatch to run_seeds.py per vision dataset and run_wine_seeds.py.

    Sec. IV-A of the paper states the stabilized oracle (multi-draw) is used
    for every vision-system number reported. run_seeds.py exposes this via
    --oracle-n-draws (default 1 = original single-draw oracle if unset); this
    driver forwards args.oracle_n_draws so a plain `--all` run matches that
    stated protocol instead of silently reverting to n_draws=1.
    """
    phase_banner("seeded")
    for ds in args.datasets:
        seeds = args.seed_map.get(ds, args.seeds)
        out_dir = ensure_dir(os.path.join(results_root, ds))
        if ds == "wine":
            log = os.path.join(out_dir, "seeded.log")
            _run_subprocess([sys.executable, "run_wine_seeds.py"], log_path=log)
            src = "wine_results_5seed.json"
            if os.path.exists(src):
                os.replace(src, os.path.join(out_dir, "seeded.json"))
        else:
            log = os.path.join(out_dir, "seeded.log")
            _run_subprocess(
                [sys.executable, "run_seeds.py", "--dataset", ds,
                 "--seeds", str(seeds),
                 "--data-root", args.data_root,
                 "--oracle-n-draws", str(args.oracle_n_draws)]
                + (["--device", args.device] if args.device else []),
                log_path=log)
            # run_seeds.py writes seeds_<ds>.json in cwd
            src = f"seeds_{ds}.json"
            if os.path.exists(src):
                os.replace(src, os.path.join(out_dir, "seeded.json"))
            tex = f"seeds_table_{ds}.tex"
            if os.path.exists(tex):
                os.replace(tex, os.path.join(out_dir, "seeds_table.tex"))


def phase_data_eff(results_root, args):
    phase_banner("data_eff")
    for ds in args.datasets:
        if ds == "wine":  # data-efficiency isn't reported for wine in the paper
            continue
        out_dir = ensure_dir(os.path.join(results_root, ds))
        log = os.path.join(out_dir, "data_eff.log")
        _run_subprocess(
            [sys.executable, "data_efficiency_sweep.py", "--full", ds,
             "--data-root", args.data_root]
            + (["--device", args.device] if args.device else []),
            log_path=log)
        src = f"data_efficiency_{ds}.json"
        if os.path.exists(src):
            os.replace(src, os.path.join(out_dir, "data_eff.json"))
    # controlled model too
    out_dir = ensure_dir(os.path.join(results_root, "controlled"))
    log = os.path.join(out_dir, "data_eff.log")
    _run_subprocess([sys.executable, "data_efficiency_sweep.py"], log_path=log)
    src = "data_efficiency_controlled.json"
    if os.path.exists(src):
        os.replace(src, os.path.join(out_dir, "data_eff.json"))


def phase_gate(results_root, args):
    """CV(g_enc) diagnostics + gate sensitivity per dataset."""
    phase_banner("gate")
    for ds in args.datasets:
        out_dir = ensure_dir(os.path.join(results_root, ds))
        # (a) diagnostics
        log = os.path.join(out_dir, "gate_diag.log")
        _run_subprocess(
            [sys.executable, "get_gate_diagnostics.py", "--dataset", ds,
             "--data-root", args.data_root]
            + (["--device", args.device] if args.device else []),
            log_path=log)
        # (b) sensitivity sweep (Sec. IV-I)
        log = os.path.join(out_dir, "gate_sens.log")
        _run_subprocess(
            [sys.executable, "gate_sensitivity_sweep.py", "--dataset", ds,
             "--data-root", args.data_root]
            + (["--device", args.device] if args.device else []),
            log_path=log)

    # gate diagnostics don't currently write JSON; parse the logs to extract
    # CV(g_enc) and encoder-limited fraction into results/<ds>/gate.json
    _parse_gate_logs(results_root, args.datasets)


def _parse_gate_logs(results_root, datasets):
    """Parse CV(g_enc), enc-limited fraction, and gate decision from logs."""
    import re
    for ds in datasets:
        log_path = os.path.join(results_root, ds, "gate_diag.log")
        if not os.path.exists(log_path): continue
        with open(log_path) as f: text = f.read()
        out = {}
        m = re.search(r"encoder-limited fraction\s*:\s*([0-9.]+)", text)
        if m: out["causal_frac"] = float(m.group(1))
        m = re.search(r"CV\(g_enc\)\s*:\s*([0-9.]+)", text)
        if m: out["cv_g_enc"] = float(m.group(1))
        m = re.search(r"CV\(g_ch\)\s*:\s*([0-9.]+)", text)
        if m: out["cv_g_ch"] = float(m.group(1))
        m = re.search(r"GATE ENABLED\s*:\s*(True|False)", text)
        if m: out["enabled"] = (m.group(1) == "True")
        if out:
            write_json(os.path.join(results_root, ds, "gate.json"), out)


def phase_ablations(results_root, args):
    """Table VII on vision backends (default: cifar10 as in paper)."""
    phase_banner("ablations")
    from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG,
                                     pick_device)
    from attrib_semcom import experiments as ex
    from attrib_semcom.paper_outputs.ablations_vision import run as run_abl

    for ds in args.datasets:
        if ds == "wine":  # not reported for wine in the paper
            continue
        cfg = SUGGESTED_CONFIG[ds]
        print(f"[ablations] training {ds} at seed 0 ...")
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=ds, kind="awgn",
            data_root=args.data_root, width=cfg["width"],
            epochs=cfg["epochs"], device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=0)
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        abl = run_abl(backend, e_vals, s_vals, T=T,
                      n=args.n, target=args.target, alpha=args.alpha)
        write_json(os.path.join(results_root, ds, "ablations.json"), abl)


def phase_hyx(results_root, args):
    """Table II hypothetical sweep + ambiguous-source validation."""
    phase_banner("hyx")
    from attrib_semcom.paper_outputs.table2 import run as run_t2
    out = run_t2()
    write_json(os.path.join(results_root, "controlled", "hyx_sweep.json"), out)

    # ambiguous source (real H(Y|X) validation, supporting evidence)
    log = os.path.join(results_root, "controlled", "ambiguous_source.log")
    ensure_dir(os.path.dirname(log))
    _run_subprocess([sys.executable, "-m", "attrib_semcom.ambiguous_source"],
                    log_path=log)
    src = "ambiguous_source_results.json"
    if os.path.exists(src):
        os.replace(src, os.path.join(results_root, "controlled",
                                     "ambiguous_source.json"))


def phase_tighter_itx(results_root, args):
    """CLUB-on-clean-path evaluation on all backends (empirical decision)."""
    phase_banner("tighter_itx")
    from attrib_semcom.paper_outputs.tighter_itx import evaluate_backend
    from attrib_semcom.backends import ControlledBackend
    from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG,
                                     pick_device)

    results = {}
    # closed-form baseline (matches the prototype's numbers)
    b_cf = ControlledBackend(chan_decay=0.80)
    e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
    results["controlled"] = evaluate_backend(b_cf, e_vals, s_vals,
                                              "controlled", per_e_T=False)

    # vision backends
    for ds in [d for d in args.datasets if d in VISION_DATASETS]:
        cfg = SUGGESTED_CONFIG[ds]
        print(f"[tighter_itx] training {ds} at seed 0 ...")
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=ds, kind="awgn",
            data_root=args.data_root, width=cfg["width"],
            epochs=cfg["epochs"], device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=0)
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))
        results[ds] = evaluate_backend(backend, e_vals, s_vals, ds, per_e_T=True)

    # wine
    if "wine" in args.datasets:
        try:
            from nonvision_wine import WineJSCCBackend
            wine = WineJSCCBackend(seed=0)
            results["wine"] = evaluate_backend(wine, list(range(0,3)),
                                                list(range(0,5)), "wine",
                                                per_e_T=True)
        except ImportError:
            print("[tighter_itx] nonvision_wine not importable; skipping Wine.")

    write_json(os.path.join(results_root, "tighter_itx_all.json"), results)

    # header summary
    print("\n[tighter_itx] SUMMARY")
    print(f"  {'backend':16s}  {'trivial Σ':>10s}  {'club Σ':>10s}  {'Δ (bits)':>10s}  {'commits (half)':>16s}  helps?")
    for label, r in results.items():
        h_trivial = f"{r['trivial']['commits_half']}/{r['trivial']['n']}"
        h_club    = f"{r['club_clean']['commits_half']}/{r['club_clean']['n']}"
        print(f"  {label:16s}  {r['trivial']['mean_sigma']:>10.3f}  "
              f"{r['club_clean']['mean_sigma']:>10.3f}  "
              f"{r['tightening_bits']:>+10.3f}  "
              f"{h_trivial:>7s} -> {h_club:>7s}   {r['helps']}")


def phase_figures(results_root, args):
    """Regenerate all figures from cached JSON."""
    phase_banner("figures")
    from attrib_semcom.paper_outputs import figures as F
    fig_dir = ensure_dir(os.path.join(results_root, "figures"))
    cf_res = read_json(os.path.join(results_root, "controlled", "closed_form_res.json"))
    cf_A   = read_json(os.path.join(results_root, "controlled", "closed_form_analyze.json"))
    cf_abl = read_json(os.path.join(results_root, "controlled", "ablations.json"))

    if cf_res:
        F.fig1_sandwich(cf_res, os.path.join(fig_dir, "fig1_sandwich.pdf"))
        F.fig4_regime_scatter(cf_res, os.path.join(fig_dir, "fig4_regimes.pdf"))
        print("  wrote fig1, fig4")

    est = read_json(os.path.join(results_root, "controlled", "estimator.json"))
    if est:
        F.fig2_estimator(est, os.path.join(fig_dir, "fig2_estimator.pdf"))
        print("  wrote fig2")
    else:
        # synthesize the estimator data from closed_form_res if possible
        # (paper's Fig. 2 uses controlled-model ground truth vs decoder estimate)
        _synthesize_estimator_json(results_root)
        est = read_json(os.path.join(results_root, "controlled", "estimator.json"))
        if est:
            F.fig2_estimator(est, os.path.join(fig_dir, "fig2_estimator.pdf"))
            print("  wrote fig2 (synthesized)")

    if cf_A:
        F.fig3_metric_bars(cf_A, os.path.join(fig_dir, "fig3_metrics.pdf"), cf_abl)
        print("  wrote fig3")

    seeded = {}
    for ds in VISION_DATASETS:
        d = read_json(os.path.join(results_root, ds, "seeded.json"))
        if d: seeded[ds] = d

    if cf_A and seeded:
        F.fig5_attribution_vs_regime(cf_A, seeded,
                                       os.path.join(fig_dir, "fig5_regime.pdf"),
                                       include_stl10=("stl10" in seeded))
        F.fig6_baselines(cf_A, seeded,
                          os.path.join(fig_dir, "fig6_baselines.pdf"))
        F.fig7_benefit_vs_override(cf_A, seeded,
                                    os.path.join(fig_dir, "fig7_benefit.pdf"))
        print("  wrote fig5, fig6, fig7")


def _synthesize_estimator_json(results_root):
    """Fig 2 requires (true I_rx, label-free estimate) pairs. Generate them on
    the controlled model where ground truth is exact."""
    from attrib_semcom.backends import ControlledBackend
    from attrib_semcom import experiments as ex
    from attrib_semcom.decomposition import (apply_temperature,
                                             label_free_predictive_info)
    b = ControlledBackend(chan_decay=0.80)
    e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
    T = ex.fit_global_temperature(b, e_vals, s_vals)

    uncal, calib = [], []
    for e in e_vals:
        for s in s_vals:
            rec = b.evaluate(e, s, n=6000, seed=1000+7*e+s)
            # true Irx (available on the controlled model)
            tl = b.true_losses(e, s)
            # Irx = I(X;Y) - Lenc - Lch
            #     = H(Y) - (L_enc + L_ch) under Assumption 1
            from attrib_semcom.decomposition import HY
            true_irx = HY(b.K) - (tl["L_enc"] + tl["L_ch"])
            # label-free estimate: predictive info of decoder posterior
            pn_uncal = rec.post_noisy
            pn_cal = apply_temperature(rec.post_noisy, T)
            est_u = label_free_predictive_info(pn_uncal, b.K)
            est_c = label_free_predictive_info(pn_cal, b.K)
            uncal.append([float(true_irx), float(est_u)])
            calib.append([float(true_irx), float(est_c)])
    write_json(os.path.join(results_root, "controlled", "estimator.json"),
               dict(uncalibrated=uncal, calibrated=calib))


def phase_tables(results_root, args):
    phase_banner("tables")
    from attrib_semcom.paper_outputs import tables as T
    tab_dir = ensure_dir(os.path.join(results_root, "tables"))

    cf_A = read_json(os.path.join(results_root, "controlled", "closed_form_analyze.json"))
    cf_abl = read_json(os.path.join(results_root, "controlled", "ablations.json"))
    if cf_A:
        with open(os.path.join(tab_dir, "table1.tex"), "w") as f:
            f.write(T.emit_table1_metric_comparison(cf_A, cf_abl))
        print("  wrote table1")

    hyx = read_json(os.path.join(results_root, "controlled", "hyx_sweep.json"))
    if hyx:
        with open(os.path.join(tab_dir, "table2.tex"), "w") as f:
            f.write(T.emit_table2_hyx(hyx))
        print("  wrote table2")

    # Table III: CV(g_enc) + override outcome pooled from seeded audits
    gate_by = {}; audit_by = {}
    for ds in ("mnist","cifar10","stl10","wine"):
        g = read_json(os.path.join(results_root, ds, "gate.json"))
        if g: gate_by[ds] = g
        s = read_json(os.path.join(results_root, ds, "seeded.json"))
        if s and "override_audit" in s:
            # aggregate helped/hurt/neutral across seeds
            total = dict(overrides=0, helped=0, hurt=0, neutral=0)
            for a in s["override_audit"]:
                total["overrides"] += a.get("n_overrides", 0)
                total["helped"] += a.get("helped", 0)
                total["hurt"] += a.get("hurt", 0)
                total["neutral"] += a.get("neutral", 0)
            n = total["overrides"]
            total["harm_rate"] = total["hurt"] / n if n else 0.0
            audit_by[ds] = total
    if gate_by and audit_by:
        with open(os.path.join(tab_dir, "table3.tex"), "w") as f:
            f.write(T.emit_table3_cv_gate(gate_by, audit_by))
        print("  wrote table3")

    seeded_by = {}
    for ds in ("mnist","cifar10","stl10"):
        d = read_json(os.path.join(results_root, ds, "seeded.json"))
        if d: seeded_by[ds] = d
    if cf_A and seeded_by:
        with open(os.path.join(tab_dir, "table4.tex"), "w") as f:
            f.write(T.emit_table4_control(cf_A, seeded_by))
        print("  wrote table4")

    de_by = {}
    for ds in ("controlled","cifar10","mnist","stl10"):
        d = read_json(os.path.join(results_root, ds, "data_eff.json"))
        if d: de_by[ds] = d
    if de_by:
        with open(os.path.join(tab_dir, "table5.tex"), "w") as f:
            f.write(T.emit_table5_data_eff(de_by))
        print("  wrote table5")

    cf_abl = read_json(os.path.join(results_root, "controlled", "ablations.json"))
    if cf_abl:
        with open(os.path.join(tab_dir, "table6.tex"), "w") as f:
            f.write(T.emit_table6_ablations_controlled(cf_abl))
        print("  wrote table6")

    for ds in VISION_DATASETS:
        abl = read_json(os.path.join(results_root, ds, "ablations.json"))
        if abl:
            with open(os.path.join(tab_dir, f"table7_{ds}.tex"), "w") as f:
                f.write(T.emit_table7_ablations_cifar(abl))
            print(f"  wrote table7_{ds}")


def phase_summary(results_root, args):
    phase_banner("summary")
    from attrib_semcom.paper_outputs.summary import write_summary
    text = write_summary(results_root,
                          os.path.join(results_root, "paper_summary.txt"))
    print(text)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="End-to-end reproducer for Paper 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="Run every phase on every dataset with default seeds.")
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated list from {mnist,cifar10,stl10,wine}. "
                         "Default: all four.")
    ap.add_argument("--phases", default=None,
                    help="Comma-separated list from "
                         "{closed_form,seeded,data_eff,gate,ablations,hyx,"
                         "tighter_itx,figures,tables,summary}. Default: all.")
    ap.add_argument("--seeds", type=int, default=20,
                    help="Default seed count for MNIST/CIFAR/Wine (20).")
    ap.add_argument("--stl10-seeds", type=int, default=10,
                    help="Seed count for STL-10 (default 10 due to compute cost).")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-root", default="./results",
                    help="Cache directory (default ./results).")
    ap.add_argument("--n", type=int, default=6000,
                    help="Eval samples for closed-form/ablations phase.")
    ap.add_argument("--target", type=float, default=0.5,
                    help="Coverage target for run_program.")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--oracle-n-draws", type=int, default=5,
                    help="Forwarded to run_seeds.py --oracle-n-draws for every "
                         "vision dataset in the 'seeded' phase. The paper "
                         "(Sec. IV-A) states the stabilized oracle (n_draws>1) "
                         "is used for every vision-system number reported; "
                         "default here is 5 to match that claim. Pass 1 to "
                         "reproduce the original single-draw oracle instead. "
                         "Wine intentionally uses the single-draw oracle "
                         "(Sec. IV-N) and is unaffected by this flag.")
    args = ap.parse_args()

    if args.all:
        args.datasets = args.datasets or ",".join(ALL_DATASETS)
        args.phases = args.phases or ",".join(ALL_PHASES)
    else:
        args.datasets = args.datasets or ",".join(ALL_DATASETS)
        args.phases = args.phases or ",".join(ALL_PHASES)

    args.datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    args.phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())

    # per-dataset seed map (STL-10 gets its own count)
    args.seed_map = {d: args.seeds for d in args.datasets}
    if "stl10" in args.seed_map:
        args.seed_map["stl10"] = args.stl10_seeds

    ensure_dir(args.results_root)

    # log the invocation
    write_json(os.path.join(args.results_root, "invocation.json"),
               dict(datasets=list(args.datasets),
                     phases=list(args.phases),
                     seeds=args.seeds,
                     stl10_seeds=args.stl10_seeds,
                     data_root=args.data_root,
                     device=args.device,
                     started=time.strftime("%Y-%m-%d %H:%M:%S")))

    print("=" * 72)
    print("run_all.py")
    print(f"  datasets : {args.datasets}")
    print(f"  phases   : {args.phases}")
    print(f"  seeds    : {args.seed_map}")
    print(f"  results  : {args.results_root}")
    print("=" * 72)

    dispatch = {
        "closed_form": phase_closed_form,
        "seeded": phase_seeded,
        "data_eff": phase_data_eff,
        "gate": phase_gate,
        "ablations": phase_ablations,
        "hyx": phase_hyx,
        "tighter_itx": phase_tighter_itx,
        "figures": phase_figures,
        "tables": phase_tables,
        "summary": phase_summary,
    }
    for p in args.phases:
        if p not in dispatch:
            print(f"[skip] unknown phase '{p}'"); continue
        t0 = time.time()
        try:
            dispatch[p](args.results_root, args)
            print(f"  [ok] {p} in {time.time()-t0:.0f}s")
        except SystemExit:
            raise
        except Exception as e:
            import traceback
            print(f"  [FAIL] phase {p}: {e}")
            traceback.print_exc()
            print(f"  Continuing to next phase.")

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()

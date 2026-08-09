r"""Multi-seed replication with variance, plus override-set diagnostics.

Closes the two reproducibility gaps a reviewer will press hardest:
  (1) every learned-system number in the paper is currently a SINGLE run at a
      fixed seed -- this reruns training + evaluation across seeds and reports
      mean, standard deviation, and a across-seed 95% interval for every headline
      quantity;
  (2) the MNIST degradation is currently explained by assertion -- this logs, for
      every fusion override, whether it HELPED or HURT relative to the SNR prior,
      so the "few overrides misfiring" claim becomes a measured quantity.

  python run_seeds.py --dataset cifar10 --seeds 5
  python run_seeds.py --dataset mnist   --seeds 5      # the negative-result case

Emits seeds_<dataset>.json and a LaTeX row block with mean +/- std.
"""
import argparse, json, time
import numpy as np
from attrib_semcom import experiments as ex
from attrib_semcom import calibration as cal
from attrib_semcom.model import build_deepjscc_backend

# Dataset configuration comes from attrib_semcom.model.SUGGESTED_CONFIG, which
# is the single source of truth shared by every entry point. Adding a dataset
# there makes it available here with no further change.
from attrib_semcom.model import SUGGESTED_CONFIG, DATASET_SPECS, pick_device

# The per-(e,s) audit is now a single implementation in attrib_semcom.experiments,
# shared with the non-vision (Wine) driver so the seed convention and paired-noise
# comparison can never drift between datasets. See experiments.override_audit and
# experiments.eval_seed for the convention.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="cifar10",
                    help="dataset NAME; rate points/width/epochs come from "
                         "SUGGESTED_CONFIG unless overridden")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default=None, help="cpu|mps|cuda; auto if unset")
    ap.add_argument("--no-gp", action="store_true",
                    help="skip the GP-surrogate baseline (default: include it)")
    ap.add_argument("--oracle-n-draws", type=int, default=1,
                    help="average the oracle over this many independent noise "
                         "draws per action (default 1 = original single-draw "
                         "oracle). Recommended >=5 on grids where the true "
                         "action gap can be comparable to single-draw noise, "
                         "e.g. STL-10 at aggressive rate points.")
    args = ap.parse_args()
    cfg = dict(SUGGESTED_CONFIG[args.dataset])
    if args.width is not None:  cfg["width"] = args.width
    if args.epochs is not None: cfg["epochs"] = args.epochs
    # an encoder-directed action needs a trained successor, so cap e at n-1
    e_vals = list(range(0, len(cfg["rate_points"]) - 1))
    s_vals = list(range(0, 7))

    # Fail LOUDLY if the GP baseline cannot run. experiments.run_program wraps
    # fit_gp_surrogate in try/except and silently drops the baseline on failure,
    # which would make it vanish from the tables with no error. Check first.
    if not args.no_gp:
        try:
            import sklearn  # noqa: F401
            from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: F401
        except Exception as e:
            raise SystemExit(
                f"scikit-learn is required for the GP-surrogate baseline ({e}).\n"
                "Install it (pip install scikit-learn) or pass --no-gp to run "
                "without the baseline.")

    keys = ["raw", "calibrated", "fused", "channel_snr", "gp_surrogate",
            "oracle_gain", "override_frac", "err_fused", "err_snr", "err_gp"]
    acc = {k: [] for k in keys}
    audits = []

    for seed in range(args.seeds):
        t0 = time.time()
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
            data_root=args.data_root,
            width=cfg["width"], epochs=cfg["epochs"], device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=seed)
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        res = ex.run_program(backend, e_vals, s_vals, T=T, target=0.5,
                             oracle_n_draws=args.oracle_n_draws)
        tgt = float(np.quantile([r["base_err0"] for r in res["rows"]], 0.75))
        res["target"] = tgt
        A = ex.analyze(res)
        acc["raw"].append(A["Q2"]["accuracy"])
        acc["calibrated"].append(A["SOTA"]["calibrated"])
        acc["fused"].append(A["SOTA"]["fused_snr"])
        acc["channel_snr"].append(A["SOTA"]["channel_snr"])
        acc["gp_surrogate"].append(A["SOTA"].get("gp_surrogate", float("nan")))
        acc["err_gp"].append(
            A["Q3"]["policies"].get("gp_surrogate", {}).get("mean", float("nan")))
        acc["oracle_gain"].append(A["Q3"]["oracle_gain_captured"])
        acc["override_frac"].append(A["Q3"]["fusion_override_fraction"])
        acc["err_fused"].append(A["Q3"]["policies"]["fused_snr"]["mean"])
        acc["err_snr"].append(A["Q3"]["policies"]["channel_snr"]["mean"])
        maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)
        audits.append(ex.override_audit(backend, e_vals, s_vals, T, maps))
        gp_s = acc["gp_surrogate"][-1]
        gp_txt = "gp=n/a " if gp_s != gp_s else f"gp={gp_s:.3f} "
        print(f"[seed {seed}] raw={acc['raw'][-1]:.3f} cal={acc['calibrated'][-1]:.3f} "
              f"snr={acc['channel_snr'][-1]:.3f} {gp_txt} ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 66)
    print(f"ACROSS {args.seeds} SEEDS  ({args.dataset})")
    print("=" * 66)
    summary = {}
    for k in keys:
        v = np.array(acc[k], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            summary[k] = dict(mean=None, std=None, ci=[None, None], values=[],
                              n_valid=0)
            print(f"  {k:15s} n/a (baseline unavailable)")
            continue
        m = v.mean(); sd = v.std(ddof=1) if len(v) > 1 else 0.0
        lo, hi = ((m - 1.96 * sd / np.sqrt(len(v)), m + 1.96 * sd / np.sqrt(len(v)))
                  if len(v) > 1 else (m, m))
        summary[k] = dict(mean=float(m), std=float(sd), ci=[float(lo), float(hi)],
                          values=[float(x) for x in v], n_valid=int(len(v)))
        star = "" if len(v) == len(acc[k]) else f"  [{len(v)}/{len(acc[k])} seeds]"
        print(f"  {k:15s} {m:.3f} +/- {sd:.3f}   95% CI [{lo:.3f}, {hi:.3f}]{star}")

    # paired across-seed comparisons (same seeds => paired is the right test)
    print("\nPAIRED ACROSS-SEED COMPARISONS (n=%d seeds)" % args.seeds)
    from math import comb
    from attrib_semcom.stats import wilcoxon_signed_rank
    # Exact two-sided sign test: appropriate at these n, unlike the normal
    # approximation used by the signed-rank routine.
    def sign_p(w, n):
        if n == 0:
            return 1.0
        k = min(w, n - w)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        return min(2 * tail, 1.0)
    min_p = sign_p(args.seeds, args.seeds)
    print(f"  [power] with n={args.seeds} seeds the smallest attainable two-sided "
          f"sign-test p is {min_p:.4f};")
    print("  [power] seed-level tests therefore CANNOT establish p<0.05 here even "
          "on a clean sweep.")
    print("  [power] treat win-counts and effect sizes as the evidence; use the "
          "within-run")
    print("  [power] paired tests over operating points for significance claims.")
    def paired(a_key, b_key, label, lower_better):
        a = np.array(acc[a_key], float); b = np.array(acc[b_key], float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 2:
            print(f"  {label:34s} n/a"); return None
        a, b = a[ok], b[ok]
        d = a - b
        W, p = wilcoxon_signed_rank(a, b)
        better = int((d < 0).sum() if lower_better else (d > 0).sum())
        nz = int((d != 0).sum())
        ps = sign_p(better, nz)
        # Cohen's d_z for paired differences
        dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
        print(f"  {label:34s} mean diff={d.mean():+.4f}  wins {better}/{len(d)}  "
              f"sign p={ps:.4f}  d_z={dz:+.2f}")
        return dict(mean_diff=float(d.mean()), wins=better, n=int(len(d)),
                    sign_p=float(ps), wilcoxon_p=float(p), cohens_dz=dz)
    comps = {}
    comps["acc_fused_vs_gp"]   = paired("fused", "gp_surrogate",
                                        "action-sel: fused vs GP", False)
    comps["acc_fused_vs_snr"]  = paired("fused", "channel_snr",
                                        "action-sel: fused vs channel-SNR", False)
    comps["err_fused_vs_gp"]   = paired("err_fused", "err_gp",
                                        "error: fused vs GP (lower better)", True)
    comps["err_fused_vs_snr"]  = paired("err_fused", "err_snr",
                                        "error: fused vs channel-SNR (lower)", True)

    print("\nOVERRIDE AUDIT (pooled across seeds)")
    tot = sum(a["n_overrides"] for a in audits)
    h = sum(a["helped"] for a in audits); x = sum(a["hurt"] for a in audits)
    print(f"  overrides={tot}  helped={h}  hurt={x}  "
          f"harm rate={x/tot if tot else 0:.3f}")
    print(f"  mean per-override error delta (positive=helped): "
          f"{np.mean([a['mean_delta'] for a in audits]):+.5f}")

    out = dict(dataset=args.dataset, seeds=args.seeds, summary=summary,
               override_audit=audits, paired_comparisons=comps,
               oracle_n_draws=args.oracle_n_draws)
    json.dump(out, open(f"seeds_{args.dataset}.json", "w"), indent=2)

    with open(f"seeds_table_{args.dataset}.tex", "w") as f:
        f.write("%% mean $\\pm$ std across %d seeds (%s)\n" % (args.seeds, args.dataset))
        for k in ["raw", "calibrated", "fused", "channel_snr", "gp_surrogate",
                  "err_fused", "err_snr", "err_gp"]:
            d = summary[k]
            if d["mean"] is None:
                f.write("%s & --- \\\\\n" % k.replace("_", "-"))
            else:
                f.write("%s & $%.3f \\pm %.3f$ \\\\\n" % (k.replace("_", "-"),
                                                          d["mean"], d["std"]))
    print(f"\nwrote seeds_{args.dataset}.json and seeds_table_{args.dataset}.tex")


if __name__ == "__main__":
    main()

"""Systematic (rho_dom, rho_pos) gate-threshold matrix across ALL FOUR
systems, cross-referenced against a self-consistently-computed override
harm rate, extending gate_sensitivity_sweep.py (which sweeps one dataset at
a time against ONLY its own baseline decision, at ONE seed) to check
whether ANY threshold pair in the published grid enables/disables all four
systems correctly at once.

ADDRESSES REVIEWER MAJOR WEAKNESS #4 (pre-deployment gate not yet a
rigorously evaluated first-class result):
    "The gate (absolute dominance / positivity of g_enc) is mentioned but is
    not a rigorously evaluated first-class result with false-positive /
    false-negative rates or enable/disable accuracy across the four
    regimes."

Sec. IV-J of the paper currently checks exactly ONE threshold pair
(rho_dom=0.90, rho_pos=0.75) against all four systems and reports that it
gets STL-10 wrong. That is a single anecdote, not a systematic evaluation.
This script sweeps the SAME grid gate_sensitivity_sweep.py already defines
(rho_dom in [0.80,0.95], rho_pos in [0.65,0.85], 20 cells) across all four
systems simultaneously, and for every cell reports how many of the four
systems get the correct enable/disable decision against ground truth.

GROUND TRUTH, and why it is computed fresh rather than borrowed from
Table V: Table V's override harm rates are pooled over 20 seeds (vision) or
5 seeds (Wine), evaluated by a DIFFERENT script (run_seeds.py /
run_wine_seeds.py) than this one. Using them directly as ground truth here
would repeat the same cross-pipeline seed-mismatch pitfall already caught
twice in this audit (the controlled-model Sigma discrepancy and STL-10's
committed-set accuracy discrepancy -- see the paper's Sec. IV-F and
Sec. IV-H footnotes). Instead, this script computes the harm rate ITSELF,
on the SAME backend/T/maps/seed used for the gate computation
(experiments.override_audit, already imported by run_seeds.py for exactly
this purpose), so ground truth and gate decision are guaranteed
self-consistent. Table V's published multi-seed numbers are printed
alongside for reference, not used as the comparison target.

STATISTICAL POWER, restated honestly: this script improves rigor from "one
threshold pair checked" to "the entire published grid checked," but it does
NOT solve the underlying n=4 problem -- there are still only four systems.
A cell scoring 4/4 is consistent with a good threshold; it is not, by
itself, statistically distinguishable from a threshold that would score 4/4
by chance at this sample size. Report results accordingly.

Usage:
    python gate_threshold_matrix.py --datasets mnist,cifar10,stl10,wine
    python gate_threshold_matrix.py --datasets mnist,cifar10 --epochs 2   # quick smoke test

Output: prints the full matrix and writes gate_threshold_matrix.json.
"""
import argparse
import json
import numpy as np
from itertools import product

try:
    from nonvision_wine import WineJSCCBackend
    HAS_WINE = True
except ImportError:
    HAS_WINE = False

# Same grids as gate_sensitivity_sweep.py, kept identical so results compose
DOM_GRID = np.round(np.arange(0.80, 0.96, 0.05), 2)
POS_GRID = np.round(np.arange(0.65, 0.86, 0.05), 2)

# Table V's published multi-seed harm rates, for reference/comparison only
# -- NOT used as the ground truth this script scores against.
PUBLISHED_HARM_RATES = dict(stl10=0.143, cifar10=0.214, wine=0.091, mnist=0.810)


def build_backend(dataset, data_root, device, epochs_override, seed=0):
    if dataset == "wine":
        if not HAS_WINE:
            raise SystemExit("Wine backend not available")
        backend = WineJSCCBackend(seed=seed)
        e_vals, s_vals = list(range(0, 3)), list(range(0, 5))
        return backend, e_vals, s_vals
    from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG,
                                     DATASET_SPECS, pick_device)
    if dataset not in DATASET_SPECS:
        raise SystemExit(f"Unknown dataset {dataset}. Choose from {sorted(DATASET_SPECS)}")
    cfg = SUGGESTED_CONFIG[dataset].copy()
    if epochs_override is not None:
        cfg["epochs"] = epochs_override
    backend = build_deepjscc_backend(
        rate_points=cfg["rate_points"], dataset=dataset, kind="awgn",
        data_root=data_root, width=cfg["width"], epochs=cfg["epochs"],
        device=device or pick_device(),
        snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=seed)
    e_vals = list(range(0, len(cfg["rate_points"]) - 1))
    s_vals = list(range(0, 7))
    return backend, e_vals, s_vals


def run_one_dataset(dataset, data_root, device, epochs_override, seed=0,
                    pool_seeds=None):
    from attrib_semcom import experiments as ex, calibration as cal
    from attrib_semcom.gate import compute_gate
    from attrib_semcom.stats import clopper_pearson

    print(f"[gate_threshold_matrix] building {dataset} (seed={seed}) ...")
    backend, e_vals, s_vals = build_backend(dataset, data_root, device,
                                            epochs_override, seed=seed)
    T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
    maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)

    # The GATE DECISION is deliberately evaluated at a single seed only,
    # matching this paper's own established convention for CV(g_enc)/the
    # gate elsewhere (Table V's caption: "a single-fit diagnostic evaluated
    # at one seed, not a seed-averaged quantity"). What needs pooling is the
    # GROUND TRUTH the decision is scored against -- a single seed's
    # override count (as low as 2 for MNIST/Wine at seed 0) is too small to
    # trust as a harm-rate estimate; a single flipped point-estimate here
    # would silently invert the "correct" label for that system. Pool
    # override_audit's totals across pool_seeds independently-trained
    # backends instead, matching Table V's own multi-seed protocol.
    if pool_seeds is None:
        pool_seeds = dict(mnist=20, cifar10=20, stl10=10, wine=5).get(dataset, 5)

    print(f"[gate_threshold_matrix] pooling ground truth over {pool_seeds} seeds ...")
    total_overrides = total_helped = total_hurt = 0
    for pool_seed in range(pool_seeds):
        if pool_seed == seed:
            pb, pT, pmaps = backend, T, maps  # reuse the already-fit seed-0 objects
        else:
            pb, pe, ps = build_backend(dataset, data_root, device, epochs_override,
                                       seed=pool_seed)
            pT = ex.fit_per_e_temperature(pb, pe, ps)
            pmaps = cal.fit_action_value_maps(pb, pe, ps, pT)
        audit = ex.override_audit(pb, e_vals, s_vals, pT, pmaps, seed0=pool_seed)
        total_overrides += audit["n_overrides"]
        total_helped += audit["helped"]
        total_hurt += audit["hurt"]

    pooled_harm = total_hurt / total_overrides if total_overrides else float("nan")
    ci_lo, ci_hi = ((None, None) if total_overrides == 0
                    else clopper_pearson(total_hurt, total_overrides))

    # Diagnose grid-invariance: capture the raw statistics at the baseline
    # threshold so a constant decision across the swept grid can be
    # explained (stat far outside every swept threshold) rather than merely
    # observed.
    baseline = compute_gate(backend, e_vals, s_vals, T, maps,
                            dominance_ratio=0.90, min_positive_frac=0.75,
                            verbose=False)

    decisions = {}
    for rho_dom, rho_pos in product(DOM_GRID, POS_GRID):
        g = compute_gate(backend, e_vals, s_vals, T, maps,
                         dominance_ratio=float(rho_dom),
                         min_positive_frac=float(rho_pos), verbose=False)
        decisions[(float(rho_dom), float(rho_pos))] = bool(g["enabled"])

    return dict(dataset=dataset,
               pooled_harm_rate=float(pooled_harm),
               pooled_harm_ci=(float(ci_lo), float(ci_hi)) if ci_lo is not None else None,
               pooled_seeds=pool_seeds,
               pooled_overrides_total=int(total_overrides),
               published_harm_rate=PUBLISHED_HARM_RATES.get(dataset),
               dominance=float(baseline["dominance"]),
               positive_frac=float(baseline["positive_frac"]),
               decisions=decisions)


def score_matrix(per_dataset, should_enable_threshold=0.30):
    """For every (rho_dom, rho_pos) cell, count how many datasets get the
    correct enable/disable decision against each dataset's OWN pooled
    (multi-seed) harm rate at should_enable_threshold. Datasets whose 95%
    Clopper-Pearson interval straddles the threshold are flagged as
    UNRELIABLE ground truth and excluded from the correct-count (reported
    separately) rather than silently scored as if the label were solid."""
    datasets = list(per_dataset.keys())
    ground_truth = {}
    reliable = {}
    for ds in datasets:
        harm = per_dataset[ds]["pooled_harm_rate"]
        ci = per_dataset[ds]["pooled_harm_ci"]
        ground_truth[ds] = harm < should_enable_threshold
        reliable[ds] = (ci is None or not (ci[0] < should_enable_threshold < ci[1]))

    cell_scores = {}
    for rho_dom, rho_pos in product(DOM_GRID, POS_GRID):
        key = (float(rho_dom), float(rho_pos))
        correct = 0
        n_reliable = 0
        detail = {}
        for ds in datasets:
            decided_enable = per_dataset[ds]["decisions"][key]
            is_correct = decided_enable == ground_truth[ds]
            if reliable[ds]:
                correct += int(is_correct)
                n_reliable += 1
            detail[ds] = dict(decided_enable=decided_enable,
                             should_enable=ground_truth[ds], correct=is_correct,
                             reliable=reliable[ds])
        cell_scores[key] = dict(correct=correct, n_reliable=n_reliable,
                               n_total=len(datasets), detail=detail)
    return ground_truth, reliable, cell_scores


def print_report(per_dataset, ground_truth, reliable, cell_scores, should_enable_threshold):
    datasets = list(per_dataset.keys())
    print(f"\n{'='*78}")
    print("GATE THRESHOLD MATRIX -- pooled multi-seed ground truth")
    print(f"{'='*78}")
    print(f"  should-enable threshold: harm_rate < {should_enable_threshold}")
    for ds in datasets:
        d = per_dataset[ds]
        ci = d["pooled_harm_ci"]
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "n/a"
        rel = "reliable" if reliable[ds] else "*** CI STRADDLES THRESHOLD -- UNRELIABLE ***"
        print(f"  {ds:10s}  pooled harm={d['pooled_harm_rate']:.3f}  95% CI={ci_str}"
              f"  ({d['pooled_overrides_total']} overrides / {d['pooled_seeds']} seeds)"
              f"  published={d['published_harm_rate']}"
              f"  -> should_enable={ground_truth[ds]}  [{rel}]")
        print(f"    dominance={d['dominance']:.3f}  positive_frac={d['positive_frac']:.3f}"
              f"  (swept ranges: rho_dom in [{DOM_GRID[0]},{DOM_GRID[-1]}], "
              f"rho_pos in [{POS_GRID[0]},{POS_GRID[-1]}])")
        n_enabled = sum(d["decisions"].values())
        if n_enabled in (0, len(d["decisions"])):
            state = "ENABLED" if n_enabled else "DISABLED"
            print(f"    GRID-INVARIANT: {state} at all {len(d['decisions'])} swept cells "
                  f"-- decision does not depend on threshold choice within this range.")

    n_unreliable = sum(1 for ds in datasets if not reliable[ds])
    if n_unreliable:
        print(f"\n  {n_unreliable}/{len(datasets)} dataset(s) have ground truth too "
              f"uncertain (CI straddles the threshold) to score against. Correct-counts "
              f"below are out of the REMAINING reliable datasets only.")

    print(f"\n  Correct-decision count per (rho_dom, rho_pos) cell "
          f"(out of reliable datasets, shown as correct/reliable):")
    header = "  rho_dom \\ rho_pos | " + "  ".join(f"{p:.2f}" for p in POS_GRID)
    print(header)
    print("  " + "-" * (20 + 6 * len(POS_GRID)))
    best_cells = []
    for d in DOM_GRID:
        row_cells = []
        for p in POS_GRID:
            key = (float(d), float(p))
            c = cell_scores[key]["correct"]
            nr = cell_scores[key]["n_reliable"]
            row_cells.append(f"{c}/{nr}")
            if nr > 0 and c == nr:
                best_cells.append(key)
        print(f"     {d:.2f}          | " + "  ".join(f"{c:>4s}" for c in row_cells))

    print()
    baseline_key = (0.90, 0.75)
    if baseline_key in cell_scores:
        bc = cell_scores[baseline_key]
        print(f"  Baseline (rho_dom=0.90, rho_pos=0.75), matching Sec. IV-J's "
              f"deployed default: {bc['correct']}/{bc['n_reliable']} correct "
              f"(of reliable datasets)")
    if best_cells:
        print(f"  Cell(s) scoring correct on all reliable datasets: {best_cells}")
        print(f"  CAVEAT: with only {len(datasets)} systems total (and possibly fewer "
              f"reliable ones), a perfect-scoring cell is NOT statistically "
              f"distinguishable from a threshold that scores perfectly by chance at "
              f"this sample size. Report as a descriptive finding, not a validated "
              f"decision rule.")
    else:
        print(f"  No cell in the swept grid is correct on all reliable datasets.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="mnist,cifar10,stl10,wine",
                    help="comma-separated subset of mnist,cifar10,stl10,wine")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the gate's OWN decision (kept single-seed, "
                         "matching the paper's established convention for "
                         "CV(g_enc)/the gate)")
    ap.add_argument("--pool-seeds", type=int, default=None,
                    help="seeds to pool for GROUND-TRUTH harm rate only "
                         "(default: dataset-specific, matching Table V's own "
                         "protocol -- 20 for mnist/cifar10, 10 for stl10, 5 "
                         "for wine). This is expensive (retrains the backend "
                         "at every pooled seed); pass a small value for a "
                         "quick smoke test, understanding the resulting CI "
                         "will be wide and datasets may show as unreliable.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override SUGGESTED_CONFIG epochs for a quick smoke test")
    ap.add_argument("--should-enable-threshold", type=float, default=0.30,
                    help="harm_rate below this is treated as ground-truth "
                         "'should enable' (default 0.30, chosen to separate "
                         "STL-10/CIFAR-10/Wine from MNIST in the published "
                         "numbers; sensitivity to THIS choice is not itself "
                         "swept here -- rerun with a different value to check)")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    per_dataset = {}
    for ds in datasets:
        per_dataset[ds] = run_one_dataset(ds, args.data_root, args.device,
                                          args.epochs, seed=args.seed,
                                          pool_seeds=args.pool_seeds)

    ground_truth, reliable, cell_scores = score_matrix(per_dataset,
                                                        args.should_enable_threshold)
    print_report(per_dataset, ground_truth, reliable, cell_scores,
                args.should_enable_threshold)

    out = dict(
        datasets=datasets,
        should_enable_threshold=args.should_enable_threshold,
        per_dataset={ds: dict(pooled_harm_rate=d["pooled_harm_rate"],
                              pooled_harm_ci=d["pooled_harm_ci"],
                              pooled_seeds=d["pooled_seeds"],
                              pooled_overrides_total=d["pooled_overrides_total"],
                              published_harm_rate=d["published_harm_rate"],
                              dominance=d["dominance"],
                              positive_frac=d["positive_frac"])
                    for ds, d in per_dataset.items()},
        ground_truth=ground_truth,
        reliable=reliable,
        cell_scores={f"{k[0]},{k[1]}": v["correct"] for k, v in cell_scores.items()},
        cell_n_reliable={f"{k[0]},{k[1]}": v["n_reliable"] for k, v in cell_scores.items()},
        cell_detail={f"{k[0]},{k[1]}": v["detail"] for k, v in cell_scores.items()},
    )
    fn = "gate_threshold_matrix.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()

"""Gate sensitivity sweep for the manuscript (§IV-I).

Sweeps ρ_dom ∈ [0.80, 0.95] and ρ_pos ∈ [0.65, 0.85] and reports whether
the enable/disable decision changes.

Usage (from project root, myproject env active):
  python gate_sensitivity_sweep.py --dataset mnist
  python gate_sensitivity_sweep.py --dataset cifar10
  python gate_sensitivity_sweep.py --dataset stl10
  python gate_sensitivity_sweep.py --dataset wine   # if you have the Wine backend
"""

import argparse
import numpy as np
from itertools import product

from attrib_semcom import experiments as ex, calibration as cal
from attrib_semcom.gate import compute_gate
from attrib_semcom.model import build_deepjscc_backend, SUGGESTED_CONFIG, pick_device

# Optional: Wine backend if present
try:
    from nonvision_wine import WineJSCCBackend
    HAS_WINE = True
except ImportError:
    HAS_WINE = False


def run_sweep(backend, e_vals, s_vals, T, maps, dataset_name, seed=0):
    # Default thresholds used in the paper
    default_dom, default_pos = 0.90, 0.75

    # Sweep grids
    dom_grid = np.round(np.arange(0.80, 0.96, 0.05), 2)   # 0.80, 0.85, 0.90, 0.95
    pos_grid = np.round(np.arange(0.65, 0.86, 0.05), 2)   # 0.65, 0.70, 0.75, 0.80, 0.85

    print(f"\n=== GATE SENSITIVITY SWEEP  ({dataset_name}, seed={seed}) ===")
    print(f"  ρ_dom grid : {list(dom_grid)}")
    print(f"  ρ_pos grid : {list(pos_grid)}")
    print()

    # Baseline decision at paper defaults
    base = compute_gate(backend, e_vals, s_vals, T, maps,
                        dominance_ratio=default_dom,
                        min_positive_frac=default_pos,
                        verbose=False)
    base_enabled = base["enabled"]
    print(f"  Baseline (ρ_dom={default_dom}, ρ_pos={default_pos}) → "
          f"{'ENABLED' if base_enabled else 'DISABLED'}")
    print(f"    dominance = {base['dominance']:.3f},  positive_frac = {base['positive_frac']:.3f}")
    print()

    flips = []
    results = []
    for rho_dom, rho_pos in product(dom_grid, pos_grid):
        g = compute_gate(backend, e_vals, s_vals, T, maps,
                         dominance_ratio=float(rho_dom),
                         min_positive_frac=float(rho_pos),
                         verbose=False)
        flipped = g["enabled"] != base_enabled
        results.append((rho_dom, rho_pos, g["enabled"], flipped))
        if flipped:
            flips.append((rho_dom, rho_pos, g["enabled"]))

    # Pretty table
    print("  ρ_dom \\ ρ_pos | " + "  ".join(f"{p:.2f}" for p in pos_grid))
    print("  " + "-" * (14 + 6 * len(pos_grid)))
    for d in dom_grid:
        row = [r for r in results if r[0] == d]
        cells = []
        for _, _, en, fl in row:
            mark = "E" if en else "D"
            if fl:
                mark = mark + "*"
            cells.append(f"{mark:>4}")
        print(f"     {d:.2f}       | " + "  ".join(cells))

    print()
    print("  Legend: E = ENABLED, D = DISABLED, * = decision flipped relative to baseline")
    if not flips:
        print("  RESULT: decision is STABLE across the entire sweep (no flips).")
    else:
        print(f"  RESULT: {len(flips)} cell(s) flipped:")
        for d, p, en in flips:
            print(f"    ρ_dom={d:.2f}, ρ_pos={p:.2f} → {'ENABLED' if en else 'DISABLED'}")
    print()
    return results, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="mnist | cifar10 | stl10 | wine | ...")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override SUGGESTED_CONFIG epochs (useful for quick tests)")
    args = ap.parse_args()

    if args.dataset == "wine":
        if not HAS_WINE:
            raise SystemExit("Wine backend not available")
        print("[wine] building WineJSCCBackend ...")
        backend = WineJSCCBackend(seed=args.seed)
        e_vals = list(range(0, 3))
        s_vals = list(range(0, 5))
    else:
        from attrib_semcom.model import DATASET_SPECS
        if args.dataset not in DATASET_SPECS:
            raise SystemExit(f"Unknown dataset {args.dataset}. "
                             f"Choose from {sorted(DATASET_SPECS)}")
        cfg = SUGGESTED_CONFIG[args.dataset].copy()
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        print(f"[{args.dataset}] training {len(cfg['rate_points'])} rate points "
              f"(width={cfg['width']}, epochs={cfg['epochs']}) ...")
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"],
            dataset=args.dataset,
            kind="awgn",
            data_root=args.data_root,
            width=cfg["width"],
            epochs=cfg["epochs"],
            device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)},
            seed=args.seed,
        )
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))

    # Fit once
    T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
    maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)

    # Sweep
    run_sweep(backend, e_vals, s_vals, T, maps, args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()

    
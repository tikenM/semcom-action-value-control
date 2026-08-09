"""Run the attribution-gate diagnostic on real trained MNIST/CIFAR-10 backends.

Prints exactly the numbers needed for the manuscript and the patent:
  - encoder-limited fraction of the calibration split
  - CV of g_enc and g_ch over the calibration deficits
  - full gate decision and fired reasons

Usage (from your project root, myproject env active):
  python get_gate_diagnostics.py --dataset mnist
  python get_gate_diagnostics.py --dataset cifar10
"""
import argparse, numpy as np
from attrib_semcom import experiments as ex, calibration as cal, decomposition as dec
from attrib_semcom.gate import compute_gate
from attrib_semcom.model import build_deepjscc_backend

# single source of truth for per-dataset configuration
from attrib_semcom.model import SUGGESTED_CONFIG as CONFIGS
from attrib_semcom.model import DATASET_SPECS, pick_device

def cov(v):
    m = np.mean(v)
    return float(np.std(v) / m) if abs(m) > 1e-9 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True,
                    help="dataset NAME (not a path)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None, help="cpu|mps|cuda; auto if unset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = CONFIGS[args.dataset]
    print(f"[{args.dataset}] training {len(cfg['rate_points'])} rate points ...")
    backend = build_deepjscc_backend(
        rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
        data_root=args.data_root,
        width=cfg["width"], epochs=cfg["epochs"], device=args.device or pick_device(),
        snr_map={s: -6.0 + 3.0*s for s in range(8)}, seed=args.seed)

    e_vals = list(range(0, len(cfg["rate_points"]) - 1))
    s_vals = list(range(0, 7))
    T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
    maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)

    per_e = isinstance(T, dict)
    enc_lim = 0; total = 0; enc_defs = []; ch_defs = []
    margins = []
    for e in e_vals:
        for s in s_vals:
            Te = T[e] if per_e else T
            rec = (backend.evaluate_cal(e, s)
                   if hasattr(backend, "evaluate_cal")
                   else backend.evaluate(e, s, seed=9000+7*e+s))
            cert = dec.certified_losses(rec, backend.K, T=Te)
            L_enc = cert["Lenc_point"]; L_ch = cert["Lch_point"]
            enc_defs.append(L_enc); ch_defs.append(L_ch)
            margins.append(L_ch - L_enc)
            enc_lim += int(L_enc > L_ch); total += 1

    causal_frac = enc_lim / total
    g_enc = np.array([maps["encoder"].predict(d) for d in enc_defs])
    g_ch  = np.array([maps["power"].predict(d)   for d in ch_defs])

    print(f"\n=== {args.dataset.upper()} GATE DIAGNOSTICS (seed={args.seed}) ===")
    print(f"  calibration points       : {total}")
    print(f"  encoder-limited fraction : {causal_frac:.4f}  ({enc_lim}/{total})")
    print(f"  margin range [L_ch-L_enc]: [{min(margins):.4f}, {max(margins):.4f}]")
    print(f"  L_enc range              : [{min(enc_defs):.4f}, {max(enc_defs):.4f}]")
    print(f"  L_ch  range              : [{min(ch_defs):.4f}, {max(ch_defs):.4f}]")
    print(f"  g_enc unique values      : {sorted(set(round(x,5) for x in g_enc))}")
    print(f"  g_enc mean={np.mean(g_enc):.6f}  std={np.std(g_enc):.6f}")
    print(f"  CV(g_enc)                : {cov(g_enc):.6f}")
    print(f"  CV(g_ch)                 : {cov(g_ch):.6f}")
    print()
    g = compute_gate(backend, e_vals, s_vals, T, maps, verbose=True)
    print(f"\n  GATE ENABLED  : {g['enabled']}")
    print(f"  FIRED REASONS : {g['fired_reasons'] or 'none (gate enabled)'}")

if __name__ == "__main__":
    main()

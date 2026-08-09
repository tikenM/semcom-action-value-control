"""System-level operator-relevant metric: transmit energy per correctly
classified image, and channel-uses (latency proxy) per correctly classified
image.

Both units are derived from constants ALREADY FIXED in the main paper's
experimental protocol (Sec. IV-A):
  - each channel-directed action step (s -> s+1) is a +3 dB SNR increase
    (snr_map = {s: -6 + 3*s}), which under a fixed noise floor corresponds
    to a 10^(3/10) ~= 2x increase in per-symbol transmit power;
  - each encoder-directed action step changes k, the number of complex
    symbols per image, directly multiplying both transmit energy per image
    (k * per-symbol power) and channel uses per image (a latency proxy at
    fixed symbol rate).

For a policy pi (channel-SNR / fused / GP-surrogate / oracle), at each
operating point this script:
  1. determines the action pi takes,
  2. computes the resulting (k, per-symbol power) and hence energy per image,
  3. computes the realized error at that point,
  4. reports energy / (1 - error) = energy PER CORRECTLY CLASSIFIED image,
     and channel-uses / (1 - error) = a latency-flavored analogue.

Usage:
    python system_metric.py --dataset cifar10
    python system_metric.py                     # controlled model (power axis only;
                                                  # no k-symbol dimension exists there)
"""
import argparse
import json
import numpy as np


def db_of_s(s):
    return -6.0 + 3.0 * s


def linear_power(s, ref_s=0):
    return 10 ** ((db_of_s(s) - db_of_s(ref_s)) / 10.0)


def run_vision(backend, e_vals, s_vals, k_of_e, T, maps, gamma_thresh, n=6000):
    from attrib_semcom import controllers as ctl, conformal as cf

    per_e = isinstance(T, dict)
    policies = {"channel_snr": [], "fused": []}
    rows = []
    for e in e_vals:
        T_e = T[e] if per_e else T
        for s in s_vals:
            seed = 101 * e + s
            act_snr = ctl.signal_channel_snr(backend, e, s, gamma_thresh)
            act_fused, _ = ctl.fused_snr_action(
                backend, e, s, n, seed, T_e, maps, gamma_thresh)

            for pname, act in [("channel_snr", act_snr), ("fused", act_fused)]:
                if act == "power":
                    e_r, s_r = e, min(s + 1, max(s_vals))
                else:
                    e_r, s_r = min(e + 1, max(e_vals)), s
                rec = ctl._after(backend, e, s, act, n, seed)
                err = cf.error_rate(rec)
                k = k_of_e[e_r]
                energy = k * linear_power(s_r)
                channel_uses = k
                policies[pname].append(dict(
                    e=e, s=s, action=act, err=err,
                    energy=energy, channel_uses=channel_uses))

    summary = {}
    for pname, pts in policies.items():
        errs = np.array([p["err"] for p in pts])
        energies = np.array([p["energy"] for p in pts])
        uses = np.array([p["channel_uses"] for p in pts])
        mean_acc = 1 - errs.mean()
        mean_energy = energies.mean()
        mean_uses = uses.mean()
        summary[pname] = dict(
            mean_error=float(errs.mean()),
            mean_accuracy=float(mean_acc),
            mean_energy_relative=float(mean_energy),
            mean_channel_uses=float(mean_uses),
            energy_per_correct=float(mean_energy / mean_acc) if mean_acc > 0 else None,
            channel_uses_per_correct=float(mean_uses / mean_acc) if mean_acc > 0 else None,
        )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "stl10"])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of training seeds to average over (default 5). "
                         "Single-seed numbers are noisy at the scale of the effects "
                         "this metric reports; do not treat n_seeds=1 output as a "
                         "paper-ready result.")
    args = ap.parse_args()

    from attrib_semcom.model import (build_deepjscc_backend, SUGGESTED_CONFIG, pick_device)
    from attrib_semcom import experiments as ex, calibration as cal

    cfg = SUGGESTED_CONFIG[args.dataset]
    e_vals = list(range(0, len(cfg["rate_points"]) - 1))
    s_vals = list(range(0, 7))
    k_of_e = {e: cfg["rate_points"][e] for e in range(len(cfg["rate_points"]))}

    print(f"SYSTEM-LEVEL METRIC  ({args.dataset}, {args.seeds} seeds)")
    print("=" * 66)

    per_seed = []
    for seed in range(args.seeds):
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
            data_root=args.data_root, width=cfg["width"], epochs=cfg["epochs"],
            device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=seed)
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)
        gamma_thresh = float(np.median([backend.gamma(s) for s in s_vals]))
        summary = run_vision(backend, e_vals, s_vals, k_of_e, T, maps, gamma_thresh)
        per_seed.append(summary)
        print(f"  [seed {seed}] fused energy_per_correct="
              f"{summary['fused']['energy_per_correct']:.2f}  "
              f"channel_snr energy_per_correct="
              f"{summary['channel_snr']['energy_per_correct']:.2f}")

    # aggregate mean +/- std across seeds, per policy per metric
    agg = {}
    for pname in ("channel_snr", "fused"):
        agg[pname] = {}
        for metric in per_seed[0][pname]:
            vals = np.array([s[pname][metric] for s in per_seed])
            agg[pname][metric] = dict(mean=float(vals.mean()),
                                      std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                                      values=[float(v) for v in vals])

    print()
    print("AGGREGATE (mean +/- std across seeds):")
    for pname in ("channel_snr", "fused"):
        e_pc = agg[pname]["energy_per_correct"]
        u_pc = agg[pname]["channel_uses_per_correct"]
        print(f"  {pname}: energy_per_correct={e_pc['mean']:.2f}+/-{e_pc['std']:.2f}  "
              f"channel_uses_per_correct={u_pc['mean']:.2f}+/-{u_pc['std']:.2f}")

    fn = f"system_metric_{args.dataset}_seedavg.json"
    json.dump(dict(dataset=args.dataset, seeds=args.seeds, per_seed=per_seed,
                   aggregate=agg), open(fn, "w"), indent=2)
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()

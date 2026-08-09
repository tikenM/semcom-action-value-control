"""V1: does run_program's Sigma match tighter_itx.py's inline Sigma?

Trains a single-seed CIFAR-10 backend the same way run_all.py does, calls
run_program on the same operating grid, and prints the per-point Sigma stats
that certified_losses (via run_program) computes.

Expected:
    trivial-bound mean Sigma ~= 2.21 bits on CIFAR-10 (matches results/tighter_itx_all.json)
    trivial-bound min Sigma  ~= 1.86 bits
    trivial-bound max Sigma  ~= 2.6 bits ish

If mean returns 2.21 (within a few percent), V1 passes and the tighter_itx
result is on the same numerical footing as the paper's certified rule.
If it comes back very different, my inline derivation has a subtle bug.

Usage:
    python verify_sigma_v1.py
"""
import numpy as np
from attrib_semcom import experiments as ex
from attrib_semcom.model import build_deepjscc_backend, SUGGESTED_CONFIG, pick_device


def main():
    ds = "cifar10"
    cfg = SUGGESTED_CONFIG[ds]
    print(f"[V1] training {ds} at seed 0 ...")
    backend = build_deepjscc_backend(
        rate_points=cfg["rate_points"], dataset=ds, kind="awgn",
        data_root="./data", width=cfg["width"], epochs=cfg["epochs"],
        device=pick_device(),
        snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=0)

    e_vals = list(range(0, len(cfg["rate_points"]) - 1))
    s_vals = list(range(0, 7))
    T = ex.fit_per_e_temperature(backend, e_vals, s_vals)

    # trivial bound (paper's default: upper_mode='dpi_capacity' -> I_tx_hi = H(Y))
    res = ex.run_program(backend, e_vals, s_vals, T=T, target=0.5)
    sigmas = np.array([r["Sigma"] for r in res["rows"]])
    print(f"[V1] CIFAR-10 run_program Sigma (trivial I_tx_hi = H(Y)):")
    print(f"     min  = {sigmas.min():.3f}")
    print(f"     mean = {sigmas.mean():.3f}")
    print(f"     max  = {sigmas.max():.3f}")
    print(f"     tighter_itx.json trivial mean_sigma was: 2.210")
    print(f"     tighter_itx.json trivial min_sigma  was: 1.856")
    print()
    print(f"[V1] verdict: {'PASS (means match within 5%)' if abs(sigmas.mean() - 2.210) < 0.11 else 'INVESTIGATE (means differ substantially)'}")


if __name__ == "__main__":
    main()

"""Table VII driver: run the paper's ablation suite on a trained vision backend.

The existing attrib_semcom.ablations module operates on any backend that
exposes the ControlledBackend interface. This driver composes the same
ablation functions on a trained DeepJSCC backend and reports:

  1. temperature calibration off/on (raw diagnosis + action-value rule)
  2. upper-bound choice: DPI+capacity vs CLUB (mean Sigma, accuracy)
  3. abstention-scale sweep (committed fraction at scales 0 / 0.5 / 1)
  4. decomposition vs decoder confidence (accuracy on the same grid)
  5. decomposition vs channel-SNR (accuracy on the same grid)
  6. raw rule vs isotonic action-value rule (isolates the mis-scaling fix)

This is one seed (typically seed 0). The paper (Table VII caption) notes
single-seed values differ from five-seed means by less than one standard
deviation, so a single-seed run is representative.
"""
import json
import numpy as np
from attrib_semcom import experiments as ex
from attrib_semcom import ablations as ab
from attrib_semcom import controllers as ctl
from attrib_semcom import calibration as cal


def _acc_agree_with_oracle(actions, oracle_actions):
    def cls(a): return "channel_limited" if a == "power" else "encoder_limited"
    return float(np.mean([cls(a) == cls(o) for a, o in zip(actions, oracle_actions)]))


def run(backend, e_vals, s_vals, T=None, n=6000, target=0.5, alpha=0.10):
    """Return a dict of ablation results structured to match Table VII rows."""
    if T is None:
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
    kw = dict(n=n, target=target, alpha=alpha)

    out = {}

    # (1) temperature calibration off/on -- affects raw AND action-value rule
    # We evaluate BOTH under uncalibrated (T=1.0) and calibrated (T=fitted).
    calib = {}
    for lbl, temp in [("uncalibrated", 1.0), ("calibrated", T)]:
        res = ex.run_program(backend, e_vals, s_vals, T=temp, **kw)
        A = ex.analyze(res)
        # raw = point-diagnosis rule (Q2 accuracy); action-value = calibrated rule
        # (SOTA["calibrated"] is action-selection accuracy under the isotonic maps)
        calib[lbl] = dict(raw_acc=float(A["Q2"]["accuracy"]),
                          action_value_acc=float(A["SOTA"]["calibrated"]))
    out["temperature_calibration"] = calib

    # (2) upper-bound choice: DPI+capacity vs CLUB (received side)
    out["upper_bound"] = ab.ablation_upper_bound(backend, e_vals, s_vals, **kw)

    # (3) abstention scale sweep
    out["abstain_sweep"] = ab.ablation_abstain_sweep(
        backend, e_vals, s_vals, scales=(0.0, 0.5, 1.0), **kw)

    # (4/5) decomposition vs confidence AND vs channel-SNR
    v = ab.ablation_confidence_vs_decomposition(backend, e_vals, s_vals, **kw)
    out["confidence_vs_decomposition"] = v

    # (6) raw rule vs isotonic action-value rule at fitted T
    #     (this is the row that isolates mis-scaling: same information both ways)
    res = ex.run_program(backend, e_vals, s_vals, T=T, **kw)
    A = ex.analyze(res)
    out["raw_vs_action_value"] = dict(
        raw_acc=float(A["Q2"]["accuracy"]),
        action_value_acc=float(A["SOTA"]["calibrated"]))

    return out

"""V2: is CLUB-on-clean-path actually an upper bound on I(Z_tx; Y)?

The controlled model exposes ground-truth mutual information at every
operating point via ControlledBackend.true_losses. This script compares
club_upper(post_clean, K) against the true I_tx at every (e, s) point.
If CLUB < true I_tx at any point, CLUB is NOT a valid upper bound, and
the tighter-Itx phase's "commits at accuracy 1.000" would be committing on
points the paper's certified guarantee should not commit on.

Reports:
    n              total operating points
    violations     points where CLUB < true I_tx (each is a bound violation)
    min_slack      min(CLUB - true I_tx), signed; negative = violation
    mean_slack     mean(CLUB - true I_tx); positive = slack, negative = bias
    max_violation  the largest (in magnitude) violation, if any

If violations = 0, the bound is empirically valid on the controlled model
and the tighter_itx result is scientifically defensible. If violations > 0,
we STOP: the whole result must be dropped.
"""
import numpy as np
from attrib_semcom.backends import ControlledBackend
from attrib_semcom.decomposition import (apply_temperature, club_upper, HY)
from attrib_semcom import experiments as ex


def run(e_vals=None, s_vals=None):
    if e_vals is None: e_vals = list(range(0, 6))
    if s_vals is None: s_vals = list(range(0, 8))

    b = ControlledBackend(chan_decay=0.80)
    T = ex.fit_global_temperature(b, e_vals, s_vals)
    H = HY(b.K)

    rows = []
    for e in e_vals:
        for s in s_vals:
            rec = b.evaluate(e, s, n=6000, seed=1000 + 7*e + s)
            # true I_tx from ground-truth losses (Assumption 1: I(X;Y)=H(Y),
            # so I_tx = H(Y) - L_enc)
            tl = b.true_losses(e, s)
            true_itx = H - tl["L_enc"]
            # CLUB on clean-path posteriors, at calibrated temperature
            pc = apply_temperature(rec.post_clean, T)
            club_itx = club_upper(pc, b.K)
            slack = club_itx - true_itx    # positive => upper bound holds
            rows.append(dict(e=e, s=s, true_itx=float(true_itx),
                             club_itx=float(club_itx),
                             slack=float(slack),
                             violates=bool(slack < 0)))

    slacks = np.array([r["slack"] for r in rows])
    violations = int(np.sum(slacks < 0))
    n = len(rows)

    print("=" * 66)
    print("V2: CLUB-on-clean-path validity check (controlled model)")
    print("=" * 66)
    print(f"  n operating points: {n}")
    print(f"  violations (CLUB < true I_tx): {violations}/{n}")
    print(f"  min slack (CLUB - true I_tx): {slacks.min():+.4f} bits")
    print(f"  mean slack: {slacks.mean():+.4f} bits")
    print(f"  max slack: {slacks.max():+.4f} bits")
    if violations > 0:
        worst = min(rows, key=lambda r: r["slack"])
        print(f"  WORST violation at e={worst['e']}, s={worst['s']}: "
              f"true_itx={worst['true_itx']:.4f}, club_itx={worst['club_itx']:.4f}")
        print()
        print("  VERDICT: BOUND IS NOT VALID. The tighter-Itx result must be")
        print("  DROPPED from the paper. CLUB(post_clean) does not upper-bound")
        print("  I(Z_tx; Y) on the controlled model, so the 'commits at 1.000'")
        print("  numbers are committing on points where the guarantee does not")
        print("  hold. Return to the trivial H(Y) bound and keep the future-work")
        print("  framing.")
    else:
        print()
        print("  VERDICT: bound holds on all controlled-model points.")
        print("  The tighter-Itx result is empirically valid on the model where")
        print("  we can check it. This does NOT prove validity on MNIST/CIFAR/")
        print("  STL10 (no ground truth there), but it does mean CLUB-on-clean-")
        print("  path is a defensible upper-bound choice under the same")
        print("  calibration assumption the paper already makes for CLUB on the")
        print("  received side.")
    print()
    return dict(n=n, violations=violations,
                min_slack=float(slacks.min()),
                mean_slack=float(slacks.mean()),
                max_slack=float(slacks.max()),
                rows=rows)


if __name__ == "__main__":
    import json
    out = run()
    json.dump(out, open("v2_club_validity.json", "w"), indent=2)
    print("wrote v2_club_validity.json")

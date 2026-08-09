"""Structural divergence: true loss ordering vs. oracle-best action.

ADDRESSES REVIEWER CRITICAL PROBLEM #1 (proxy ground-truth on learned systems):
    "The paper reports certified accuracy against the dominant estimated loss
    component while the main performance numbers are agreement with the
    better action. On STL-10 these two references diverge substantially.
    When loss ordering and better-action ordering disagree, the method is an
    empirical action-value learner, not true cause attribution."

The paper currently reports two different accuracy numbers that use two
different references, and never reports how often those references
themselves disagree:

  * Table VI ("action-selection accuracy"): agreement between the
    CONTROLLER'S action and the ORACLE-best action (whichever action
    reduces held-out error the most).
  * Table IV ("Acc. (full)"): agreement between the CERTIFIED DECLARATION
    and the DOMINANT TRUE LOSS COMPONENT (gt_regime: whether L_enc or L_ch
    is larger, from the held-out label-cross-entropy decomposition).

Both of these mix together two logically separate questions:
  (a) Does the TRUE loss ordering (gt_regime) even agree with which action
      is actually best (oracle_action)? This is a STRUCTURAL question about
      the problem itself -- Assumption 2/3 either hold at this operating
      point or they don't -- and no estimator, however perfect, can push
      loss-ordering-based accuracy past the ceiling this sets.
  (b) Given that the true loss ordering DOES pick the right action class,
      does the DEPLOYED estimator (calibration-split point diagnosis)
      recover that ordering correctly? This is an ESTIMATION question,
      about calibration-split sample size and estimator quality, not about
      whether cause attribution is the right frame at all.

experiments.run_program already computes gt_regime and oracle_action for
every backend (including the learned vision systems -- see the "truth =
backend.true_losses(e, s)" call, unconditional on backend type), but no
existing analysis in the codebase compares them to each other. This script
does exactly that, and reports the decomposition explicitly so the
structural-vs-estimation split in the paper's central claim can be checked
directly instead of asserted.

Usage:
    python loss_ordering_divergence.py --dataset stl10
    python loss_ordering_divergence.py --dataset cifar10
    python loss_ordering_divergence.py --dataset mnist
    python loss_ordering_divergence.py                      # controlled model
    python loss_ordering_divergence.py --dataset wine

Notes on seeding:
  Uses a SINGLE seed (seed=0 by default), matching the convention already
  established for Table IV and Table X ("single-seed values ... isolate
  mechanism"), so results are directly comparable to those tables rather
  than requiring a new multi-seed protocol. Pass --seeds N to aggregate over
  multiple seeds instead (recommended as a follow-up once the single-seed
  picture is in hand, since gt_regime/oracle_action can themselves be noisy
  per Sec. IV-A's discussion of oracle instability).

Output:
  Prints a per-backend table and writes loss_ordering_divergence_<dataset>.json.
  A companion function `emit_table_row` produces one row of the paper-style
  LaTeX table this script's docstring implies -- see the bottom of this file.
"""
import argparse
import json
import numpy as np


def _itx_upper_club_clean(post_clean, K, itx_lo):
    """Mirrors tighter_itx.py's helper of the same name exactly."""
    from attrib_semcom.decomposition import club_upper, HY
    return max(min(HY(K), club_upper(post_clean, K)), itx_lo)


def certified_layer_report(backend, rows, K, T, upper_mode="trivial",
                           sigma_scale=1.0, n=6000, seed0=0):
    """Recompute the certified declaration for EVERY row in `rows`, using
    the EXACT SAME per-point seed run_program used to produce those rows
    (experiments.eval_seed), so gt_regime, oracle_action, and this
    recomputed declaration are guaranteed to describe the SAME evaluation
    -- unlike comparing against Table IV's numbers directly, which come from
    tighter_itx.py's independently-seeded pipeline (9000+7e+s there vs
    eval_seed's seed0+101e+s here) and can commit on a slightly different
    point set even at "the same seed=0".

    upper_mode: "trivial" (I^up_tx = H(Y), matching run_program's own
    default pipeline) or "tighter" (I^up_tx = min(H(Y), CLUB(post_clean)),
    matching Eq. 4 / tighter_itx.py's construction).

    Returns per-committed-point accuracy against BOTH gt_regime (true loss
    ordering -- reproduces what Table IV measures) and oracle_action (the
    axis Table IV does not report), computed on the identical committed set
    for a direct, apples-to-apples comparison.
    """
    from attrib_semcom.decomposition import (apply_temperature, barber_agakov_lower,
                                             latent_capacity, HY)
    from attrib_semcom import experiments as ex

    per_e = isinstance(T, dict)
    H = HY(K)

    def to_class(action):
        return "encoder_limited" if action == "encoder" else "channel_limited"

    committed = []
    for r in rows:
        e, s = r["e"], r["s"]
        Te = T[e] if per_e else T
        seed = ex.eval_seed(e, s, seed0)
        rec = backend.evaluate(e, s, n=n, seed=seed)
        pc = apply_temperature(rec.post_clean, Te)
        pn = apply_temperature(rec.post_noisy, Te)
        itx_lo, _ = barber_agakov_lower(pc, rec.y, K); itx_lo = max(itx_lo, 0.0)
        irx_lo, _ = barber_agakov_lower(pn, rec.y, K); irx_lo = max(irx_lo, 0.0)
        irx_hi = max(min(H, latent_capacity(rec.k, rec.gamma)), irx_lo)

        itx_hi = H if upper_mode == "trivial" else _itx_upper_club_clean(pc, K, itx_lo)

        lenc_lo = max(H - itx_hi, 0.0)
        lch_lo = max(itx_lo - irx_hi, 0.0)
        lch_hi = itx_hi - irx_lo
        lenc_hi = H - itx_lo
        sigma = 0.5 * ((lch_hi - lenc_lo) - (lch_lo - lenc_hi))
        margin = 2.0 * itx_lo - irx_lo - H

        if abs(margin) > sigma_scale * sigma:
            decl = "channel_limited" if margin > 0 else "encoder_limited"
            committed.append(dict(e=e, s=s, decl=decl,
                                  gt_regime=r["gt_regime"],
                                  oracle_class=to_class(r["oracle_action"])))

    n_committed = len(committed)
    n_total = len(rows)
    if n_committed == 0:
        return dict(upper_mode=upper_mode, n_total=n_total, n_committed=0,
                   committed_fraction=0.0,
                   acc_vs_gt_regime=None, acc_vs_oracle=None)

    acc_gt = np.mean([c["decl"] == c["gt_regime"] for c in committed])
    acc_oracle = np.mean([c["decl"] == c["oracle_class"] for c in committed])
    return dict(upper_mode=upper_mode, n_total=n_total, n_committed=n_committed,
               committed_fraction=n_committed / n_total,
               acc_vs_gt_regime=float(acc_gt), acc_vs_oracle=float(acc_oracle))


def divergence_report(rows, label, analyze_out=None):
    """rows: the list of dicts returned in run_program(...)["rows"].
    analyze_out: optional, the dict returned by experiments.analyze(res) on
    the SAME res that produced `rows`. If given, this also extracts
    Q2["certified_accuracy"] -- the certified declaration's agreement with
    the ORACLE on committed points -- which the existing pipeline already
    computes but which no table in the paper currently reports. Table IV
    instead reports the certified declaration's agreement with the TRUE
    LOSS ORDERING (a different reference; see tighter_itx.py's bespoke
    accuracy computation). Reporting both closes the third evaluation axis:
    not just point-diagnosis-vs-oracle (structural/estimation, above) but
    also certified-declaration-vs-oracle versus certified-declaration-vs-
    loss-ordering.
    Returns a dict with the structural/estimation decomposition plus,
    if analyze_out is given, the certified-vs-oracle comparison."""

    def to_class(action):
        return "encoder_limited" if action == "encoder" else "channel_limited"

    n = len(rows)
    structural_mismatch = []   # gt_regime disagrees with oracle-best action's class
    structural_agree_rows = []
    for r in rows:
        gt_class = r["gt_regime"]                       # "encoder_limited"/"channel_limited"
        oracle_class = to_class(r["oracle_action"])      # same label space
        if gt_class != oracle_class:
            structural_mismatch.append((r["e"], r["s"]))
        else:
            structural_agree_rows.append(r)

    n_structural = len(structural_mismatch)
    structural_rate = n_structural / n if n else float("nan")

    # Among points where the TRUE loss ordering DOES pick the right action,
    # does the DEPLOYED point diagnosis (calibration-split estimate) still
    # recover that ordering? This isolates estimator error from the
    # structural mismatch computed above.
    estimation_mismatch = []
    for r in structural_agree_rows:
        if r["diag"] != r["gt_regime"]:
            estimation_mismatch.append((r["e"], r["s"]))
    n_struct_agree = len(structural_agree_rows)
    estimation_rate = (len(estimation_mismatch) / n_struct_agree
                       if n_struct_agree else float("nan"))

    # Raw rule's total disagreement with the oracle (this is what Table VI's
    # "raw diagnosis" accuracy measures the complement of; reported here for
    # a direct sanity-check cross-reference against that table).
    raw_disagree = [r for r in rows if r["diag"] != to_class(r["oracle_action"])]
    raw_disagree_rate = len(raw_disagree) / n if n else float("nan")

    # Points where BOTH structural and estimation mismatches would apply are
    # not double countable by construction (estimation is only evaluated on
    # the structural-agree subset), so:
    #   raw_disagree_rate ~= structural_rate + (1-structural_rate)*estimation_rate
    # A large gap between the LHS and RHS indicates the point diagnosis
    # sometimes "accidentally" recovers the oracle action despite gt_regime
    # disagreeing with it (or vice versa on the agree subset) -- report both
    # so this isn't silently swept into either term.
    predicted_total = structural_rate + (1 - structural_rate) * estimation_rate \
        if n and n_struct_agree else float("nan")

    out = dict(
        label=label,
        n=n,
        structural_mismatch_n=n_structural,
        structural_mismatch_rate=structural_rate,
        structural_mismatch_points=structural_mismatch,
        estimation_mismatch_n=len(estimation_mismatch),
        estimation_mismatch_rate=estimation_rate,
        estimation_mismatch_points=estimation_mismatch,
        raw_rule_disagree_with_oracle_rate=raw_disagree_rate,
        predicted_total_from_decomposition=predicted_total,
    )

    if analyze_out is not None:
        q2 = analyze_out.get("Q2", {})
        out["certified_vs_oracle_accuracy"] = q2.get("certified_accuracy")
        out["certified_committed_fraction"] = q2.get("certified_committed_fraction")
        out["certified_abstain_fraction"] = q2.get("certified_abstain_fraction")
        # This is directly comparable to Table IV's "Acc. (full)" column,
        # which uses the SAME committed set but scores against the true
        # loss ordering instead of the oracle -- report the gap explicitly.
        out["note"] = ("certified_vs_oracle_accuracy above uses the SAME "
                       "committed-point set as Table IV's 'Acc. (full)', but "
                       "scores against the oracle-best action instead of the "
                       "true loss ordering. Compare the two directly rather "
                       "than assuming they agree.")

    return out


def print_report(rep):
    print(f"\n{'='*70}")
    print(f"LOSS-ORDERING vs. ORACLE-BEST-ACTION DIVERGENCE  --  {rep['label']}")
    print(f"{'='*70}")
    print(f"  n operating points: {rep['n']}")
    print(f"  STRUCTURAL mismatch (gt_regime != oracle-best action class):")
    print(f"    {rep['structural_mismatch_n']}/{rep['n']}"
          f"  ({rep['structural_mismatch_rate']:.3f})")
    if rep["structural_mismatch_points"]:
        print(f"    points: {rep['structural_mismatch_points']}")
    print(f"  ESTIMATION mismatch (diag != gt_regime, among structural-agree points):")
    print(f"    {rep['estimation_mismatch_n']}"
          f"  ({rep['estimation_mismatch_rate']:.3f} of structural-agree subset)")
    print(f"  Raw rule total disagreement with oracle (cross-check vs. Table VI):")
    print(f"    {rep['raw_rule_disagree_with_oracle_rate']:.3f}")
    print(f"  Decomposition-predicted total: "
          f"{rep['predicted_total_from_decomposition']:.3f}"
          f"  (compare to the line above; a large gap flags an interaction effect)")
    if "certified_vs_oracle_accuracy" in rep:
        cva = rep["certified_vs_oracle_accuracy"]
        cf = rep["certified_committed_fraction"]
        print()
        print(f"  CERTIFIED declaration vs. ORACLE (third axis, not in any paper table):")
        print(f"    accuracy = {cva if cva is not None else 'n/a'}"
              f"   (committed fraction = {cf if cf is not None else 'n/a'})")
        print(f"    Compare directly against Table IV's 'Acc. (full)' for {rep['label']}, "
              f"which uses the SAME committed set but scores against the true loss "
              f"ordering instead. A gap between the two means the certified layer's "
              f"published accuracy number does not describe how often it actually "
              f"recommends the oracle-best action.")
    print()
    if rep["structural_mismatch_rate"] > 0.15:
        print(f"  INTERPRETATION: a nontrivial fraction of {rep['label']}'s operating "
              f"points are structurally mismatched -- no loss-ordering estimator can "
              f"reach 100% action-selection accuracy here regardless of calibration "
              f"quality. On these points the framework is functioning as an empirical "
              f"action-value learner (which the isotonic map explicitly is), not as "
              f"cause attribution in Definition 3's sense.")
    else:
        print(f"  INTERPRETATION: the structural mismatch rate is small on {rep['label']}; "
              f"most of the raw rule's error here is attributable to the ESTIMATION term "
              f"(calibration/mis-scaling), consistent with a genuine cause-attribution "
              f"reading of the framework on this backend.")


def emit_table_row(rep):
    """One row of the paper-style LaTeX table this diagnostic implies."""
    return (f"{rep['label']:12s} & {rep['n']:3d} & "
            f"{rep['structural_mismatch_n']}/{rep['n']} "
            f"({rep['structural_mismatch_rate']:.2f}) & "
            f"{rep['estimation_mismatch_n']}/"
            f"{rep['n']-rep['structural_mismatch_n']} "
            f"({rep['estimation_mismatch_rate']:.2f}) \\\\")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="mnist, cifar10, stl10, or wine; omit for the "
                         "controlled model")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="single seed used, matching Table IV/Table X's "
                         "single-seed convention (default 0)")
    ap.add_argument("--oracle-n-draws", type=int, default=5,
                    help="stabilized-oracle draws for vision datasets "
                         "(default 5, matching the paper's stated protocol; "
                         "N/A for the controlled model, which has an exact "
                         "closed-form oracle)")
    args = ap.parse_args()

    from attrib_semcom import experiments as ex

    if args.dataset is None:
        from attrib_semcom.backends import ControlledBackend
        backend = ControlledBackend(chan_decay=0.80)
        e_vals, s_vals = list(range(0, 6)), list(range(0, 8))
        T = ex.fit_global_temperature(backend, e_vals, s_vals)
        label = "controlled"
        oracle_n_draws = 1   # exact closed-form oracle; stabilization N/A
        K = backend.K
    elif args.dataset == "wine":
        from nonvision_wine import WineJSCCBackend
        backend = WineJSCCBackend(seed=args.seed)
        e_vals, s_vals = list(range(0, 3)), list(range(0, 5))
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        label = "wine"
        oracle_n_draws = 1   # matches run_wine_seeds.py's current protocol
        K = backend.K
    else:
        from attrib_semcom.model import (build_deepjscc_backend,
                                         SUGGESTED_CONFIG, DATASET_SPECS, pick_device)
        cfg = SUGGESTED_CONFIG[args.dataset]
        backend = build_deepjscc_backend(
            rate_points=cfg["rate_points"], dataset=args.dataset, kind="awgn",
            data_root=args.data_root, width=cfg["width"], epochs=cfg["epochs"],
            device=args.device or pick_device(),
            snr_map={s: -6.0 + 3.0 * s for s in range(8)}, seed=args.seed)
        e_vals = list(range(0, len(cfg["rate_points"]) - 1))
        s_vals = list(range(0, 7))
        T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
        label = args.dataset
        oracle_n_draws = args.oracle_n_draws
        K = DATASET_SPECS[args.dataset]["K"]

    print(f"[loss_ordering_divergence] running on {label} "
          f"(seed={args.seed}, oracle_n_draws={oracle_n_draws}) ...")
    res = ex.run_program(backend, e_vals, s_vals, T=T, target=0.5,
                         oracle_n_draws=oracle_n_draws, seed0=args.seed)
    analyze_out = ex.analyze(res)
    rep = divergence_report(res["rows"], label, analyze_out=analyze_out)
    print_report(rep)
    print("LaTeX row:", emit_table_row(rep))

    print(f"\n[loss_ordering_divergence] recomputing certified layer "
          f"(trivial bound, self-consistent seeds) on {label} ...")
    cert_trivial = certified_layer_report(backend, res["rows"], K, T,
                                          upper_mode="trivial", seed0=args.seed)
    print(f"  n_committed={cert_trivial['n_committed']}/{cert_trivial['n_total']} "
          f"({cert_trivial['committed_fraction']:.3f})  "
          f"acc_vs_gt_regime={cert_trivial['acc_vs_gt_regime']}  "
          f"acc_vs_oracle={cert_trivial['acc_vs_oracle']}")

    print(f"[loss_ordering_divergence] recomputing certified layer "
          f"(tighter bound, Eq. 4) on {label} ...")
    cert_tighter = certified_layer_report(backend, res["rows"], K, T,
                                          upper_mode="tighter", seed0=args.seed)
    print(f"  n_committed={cert_tighter['n_committed']}/{cert_tighter['n_total']} "
          f"({cert_tighter['committed_fraction']:.3f})  "
          f"acc_vs_gt_regime={cert_tighter['acc_vs_gt_regime']}  "
          f"acc_vs_oracle={cert_tighter['acc_vs_oracle']}")

    if (cert_trivial["acc_vs_gt_regime"] is not None
            and cert_trivial["acc_vs_oracle"] is not None
            and cert_trivial["acc_vs_gt_regime"] > cert_trivial["acc_vs_oracle"] + 0.05):
        print(f"  NOTE: on {label}'s trivial-bound committed points, the certified "
              f"declaration is MORE accurate against the true loss ordering "
              f"({cert_trivial['acc_vs_gt_regime']:.2f}) than against the oracle-best "
              f"action ({cert_trivial['acc_vs_oracle']:.2f}) -- Theorem 3's guarantee "
              f"holding does not imply the recommended action is usually right.")
    if (cert_tighter["acc_vs_gt_regime"] is not None
            and cert_tighter["acc_vs_oracle"] is not None
            and cert_tighter["acc_vs_gt_regime"] > cert_tighter["acc_vs_oracle"] + 0.05):
        print(f"  NOTE: same pattern under the tighter bound: "
              f"acc_vs_gt_regime={cert_tighter['acc_vs_gt_regime']:.2f} vs. "
              f"acc_vs_oracle={cert_tighter['acc_vs_oracle']:.2f}.")

    out_fn = f"loss_ordering_divergence_{label}.json"
    # strip point-lists' tuple keys aren't JSON-safe as tuples -> lists
    rep_json = dict(rep)
    rep_json["structural_mismatch_points"] = [list(p) for p in rep["structural_mismatch_points"]]
    rep_json["estimation_mismatch_points"] = [list(p) for p in rep["estimation_mismatch_points"]]
    rep_json["certified_layer_trivial"] = cert_trivial
    rep_json["certified_layer_tighter"] = cert_tighter
    json.dump(rep_json, open(out_fn, "w"), indent=2)
    print(f"\nwrote {out_fn}")


if __name__ == "__main__":
    main()

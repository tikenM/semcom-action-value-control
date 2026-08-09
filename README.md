# When Cause Attribution Fails

Code and results for *When Cause Attribution Fails: Label-Free Action-Value
Control for Task-Oriented Semantic Communication Under Discrete Encoder
Switching*.

The study decomposes task-information loss into encoder-attributable and
channel-attributable components, builds a certified data-processing (DPI)
diagnostic on that decomposition, and then tests whether correctly
identifying the dominant loss component actually identifies the best
available action. Under discrete encoder switching it frequently does not.
A label-free isotonic action-value learner, fit against realized error
reduction rather than the loss ordering, recovers action-selection
performance where the diagnostic structurally cannot.

---

## ⚠️ This snapshot is INCOMPLETE and will not run end-to-end

Several modules of the `attrib_semcom` package are **not included** in this
snapshot. Any script that imports them will fail with `ImportError` until
they are added.

**Missing package modules** (`attrib_semcom/`):

| Module | Provides | Needed by |
|---|---|---|
| `calibration.py` | `fit_action_value_maps` | `experiments`, `run_seeds`, gate scripts |
| `conformal.py` | `error_rate`, `coverage_report` | `experiments`, `controllers` |
| `stats.py` | `bootstrap_ci`, `wilcoxon_signed_rank`, `clopper_pearson`, `paired_bootstrap_diff` | `experiments`, `run_seeds`, `gate_threshold_matrix` |
| `ablations.py` | `ablation_*` functions | `run_all` (ablations phase), `ablations_vision` |
| `model.py` | `build_deepjscc_backend`, `SUGGESTED_CONFIG`, `DATASET_SPECS`, `pick_device` | every learned-system script |
| `gate.py` | `compute_gate` | `get_gate_diagnostics`, `gate_sensitivity_sweep`, `gate_threshold_matrix` |

**Missing top-level scripts:**

- `nonvision_wine.py` (`WineJSCCBackend`) — Wine pilot
- `data_efficiency_sweep.py` — `run_all --phases data_eff`
- `ambiguous_source.py` — `run_all --phases hyx`

**What does run without them:** the controlled-model path only, since
`ControlledBackend` is pure NumPy. See `attrib_semcom/run_table2_both.py`,
which depends only on `decomposition.py` and `backends.py`.

---

## Layout

```
attrib_semcom/
  backends.py             ControlledBackend (closed-form), DeepJSCCBackend interface
  decomposition.py        estimators, certified loss decomposition, Sigma, diagnosis
  experiments.py          run_program, analyze, override_audit, eval_seed
  controllers.py          policies, cause-agnostic baselines, GP-surrogate baseline
  run_table2_both.py      standalone controlled-model Table II reproducer
  paper_outputs/
    figures.py            Figs. 1-7
    tables.py             LaTeX table emitters
    table2.py             H(Y|X) sweep, controlled model
    tighter_itx.py        CLUB-on-clean-path transmitted-side bound
    ablations_vision.py   ablation suite on trained vision backends
    summary.py            paper_summary.txt

run_all.py                     end-to-end driver (all phases)
run_seeds.py                   multi-seed vision replication
run_wine_seeds.py              Wine pilot replication
loss_ordering_divergence.py    structural vs. estimation mismatch; certified-vs-oracle
hyx_sweep_tighter_itx.py       H(Y|X) sweep under the tighter transmitted-side bound
gate_threshold_matrix.py       (rho_dom, rho_pos) matrix across all systems
get_gate_diagnostics.py        CV(g_enc), gate decision per dataset
gate_sensitivity_sweep.py      single-dataset gate threshold sweep
system_metric.py               energy / channel-uses per correct classification
finite_sample_sigma.py         bootstrap CI on the estimator slack Sigma
hyx_sweep_general.py           H(Y|X) sweep, any backend
verify_club_validity.py        checks CLUB is a valid upper bound on ground truth
verify_sigma_v1.py             cross-checks Sigma between call sites
regenerate_figures_567.py      standalone regenerator for Figs. 5-7

results/                  cached JSON outputs and figure PDFs
paper/SemcomPaper2.tex    manuscript source (needs references.bib)
```

## Requirements

```bash
pip install -r requirements.txt
```

`numpy`, `scipy`, `matplotlib`, and `scikit-learn` cover the controlled-model
path. `torch` and `torchvision` are required only for the learned DeepJSCC
systems.

## Usage

Full reproduction (requires the missing modules above):

```bash
python run_all.py --all --seeds 20 --stl10-seeds 10 --oracle-n-draws 5
```

Regenerate figures and tables from cached JSON, without retraining:

```bash
python run_all.py --phases figures,tables
```

Individual analyses:

```bash
python loss_ordering_divergence.py --dataset stl10
python hyx_sweep_tighter_itx.py --dataset cifar10
python gate_threshold_matrix.py --datasets mnist,cifar10,stl10,wine
python attrib_semcom/run_table2_both.py        # controlled model, no torch
```

## Notes on seeds

- MNIST and CIFAR-10: 20 seeds. STL-10: 10 seeds. Wine: 5 seeds.
- The learned-system oracle averages five independent noise draws per action.
  `run_all.py` forwards this via `--oracle-n-draws` (default 5);
  `run_seeds.py` defaults to 1 if invoked directly, so pass the flag
  explicitly when calling it standalone.
- `run_wine_seeds.py` hardcodes 5 seeds and uses a single-draw oracle.
- Per-operating-point evaluation seeds follow `experiments.eval_seed`
  (`seed0 + 101*e + s`). `tighter_itx.py` uses its own convention
  (`9000 + 7*e + s`); results computed under the two conventions can differ
  slightly on near-boundary operating points.

## License

Not yet specified. Add a `LICENSE` file before publishing.

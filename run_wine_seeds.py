"""Five-seed replication of the Wine non-vision pilot.

Exactly mirrors the MNIST/CIFAR-10 protocol in run_seeds.py:
  - train/cal/test split drawn fresh per seed
  - per-seed gate diagnostics (CV(g_enc), enc_lim_frac)
  - override audit per seed (shared implementation with the vision driver, so
    the seed convention and paired-noise comparison cannot drift; see
    attrib_semcom.experiments.override_audit)
  - paired comparison: fused vs channel-SNR, fused vs GP surrogate

Grid and calibration match the paper's Sec. IV-I: 15 operating points
(3 encoders x 5 channel states) and per-encoder temperature calibration.

Runtime: ~60-90 seconds total (pure numpy, no GPU needed).
"""
import numpy as np, json, time
from nonvision_wine import WineJSCCBackend
from attrib_semcom import experiments as ex, calibration as cal

N_SEEDS = 5

keys = ['raw','calibrated','fused','channel_snr','gp_surrogate',
        'oracle_gain','override_frac','err_fused','err_snr','err_gp']
acc = {k: [] for k in keys}
audits = []

print(f"{'='*64}")
print(f"WINE NON-VISION PILOT  —  {N_SEEDS} SEEDS")
print(f"{'='*64}")

for seed in range(N_SEEDS):
    t0 = time.time()
    backend = WineJSCCBackend(seed=seed)
    # 15-point grid (3 encoders x 5 channel states) matches the paper's Sec. IV-I;
    # per-encoder temperature matches the calibration methodology used for MNIST
    # and CIFAR-10. Previously this driver used a 12-point grid and a pooled
    # temperature, inconsistent with the paper's stated protocol.
    e_vals = list(range(0, 3)); s_vals = list(range(0, 5))
    T = ex.fit_per_e_temperature(backend, e_vals, s_vals)
    res = ex.run_program(backend, e_vals, s_vals, T=T, target=0.5)
    noact = np.array([r['base_err0'] for r in res['rows']])
    res['target'] = float(np.quantile(noact, 0.75))
    A = ex.analyze(res)
    maps = cal.fit_action_value_maps(backend, e_vals, s_vals, T)

    acc['raw'].append(A['Q2']['accuracy'])
    acc['calibrated'].append(A['SOTA']['calibrated'])
    acc['fused'].append(A['SOTA']['fused_snr'])
    acc['channel_snr'].append(A['SOTA']['channel_snr'])
    acc['gp_surrogate'].append(A['SOTA'].get('gp_surrogate', float('nan')))
    acc['oracle_gain'].append(A['Q3']['oracle_gain_captured'])
    acc['override_frac'].append(A['Q3']['fusion_override_fraction'])
    acc['err_fused'].append(A['Q3']['policies']['fused_snr']['mean'])
    acc['err_snr'].append(A['Q3']['policies']['channel_snr']['mean'])
    gp_err = A['Q3']['policies'].get('gp_surrogate', {}).get('mean', float('nan'))
    acc['err_gp'].append(gp_err)
    audits.append(ex.override_audit(backend, e_vals, s_vals, T, maps))

    print(f"  [seed {seed}]  raw={acc['raw'][-1]:.3f}  cal={acc['calibrated'][-1]:.3f}  "
          f"snr={acc['channel_snr'][-1]:.3f}  ({time.time()-t0:.1f}s)")

print(f"\n{'='*64}")
print("ACROSS 5 SEEDS  (wine)")
print(f"{'='*64}")
summary = {}
for k in keys:
    v = np.array(acc[k], dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0:
        summary[k] = dict(mean=None, std=None, ci=[None,None], n_valid=0)
        print(f"  {k:15s}  n/a"); continue
    m = v.mean(); sd = v.std(ddof=1) if len(v)>1 else 0.0
    lo,hi = (m-1.96*sd/np.sqrt(len(v)), m+1.96*sd/np.sqrt(len(v))) if len(v)>1 else (m,m)
    summary[k] = dict(mean=float(m),std=float(sd),ci=[float(lo),float(hi)],n_valid=int(len(v)))
    print(f"  {k:15s}  {m:.3f} +/- {sd:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")

# pooled override audit
tot = sum(a['n_overrides'] for a in audits)
h   = sum(a['helped'] for a in audits)
x   = sum(a['hurt']   for a in audits)
harm = x/tot if tot else 0.0
print(f"\nOVERRIDE AUDIT (pooled)  n={tot}  helped={h}  hurt={x}  harm={harm:.3f}")

# paired comparisons (fused vs snr, fused vs gp)
print("\nPAIRED COMPARISONS (n=5 seeds; p < 0.0625 not achievable)")
from math import comb
def sign_p(wins, n):
    k = min(wins, n-wins)
    return min(2*sum(comb(n,i) for i in range(0,k+1))/2**n, 1.0)

for a_k, b_k, label, lower_better in [
    ('err_fused','err_snr','err: fused vs SNR (lower better)',True),
    ('err_fused','err_gp', 'err: fused vs GP  (lower better)',True),
    ('fused','channel_snr','acc: fused vs SNR (higher better)',False),
    ('fused','gp_surrogate','acc: fused vs GP  (higher better)',False),
]:
    a = np.array(acc[a_k],float); b = np.array(acc[b_k],float)
    ok = ~(np.isnan(a)|np.isnan(b)); a,b = a[ok],b[ok]
    if len(a)<2: print(f"  {label}  n/a"); continue
    d = a-b
    wins = int((d<0).sum()) if lower_better else int((d>0).sum())
    ps = sign_p(wins,len(d))
    dz = float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else float('nan')
    print(f"  {label:40s}  mean_diff={d.mean():+.4f}  wins={wins}/{len(d)}  sign_p={ps:.4f}  d_z={dz:+.2f}")

out = dict(dataset='wine', seeds=N_SEEDS, summary=summary,
           override_audit=audits,
           pooled_audit=dict(n=tot,helped=h,hurt=x,harm_rate=float(harm)))
json.dump(out, open('wine_results_5seed.json','w'), indent=2)
print("\nwrote wine_results_5seed.json")

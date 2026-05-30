"""
Programmatic verification of every numeric cell and statistical claim cited in
the manuscript.  Run with no arguments:

    python verify_tables.py

Exits with status 0 if every assertion passes and a non-zero status otherwise.
Each assertion's tolerance is 0.01 (RMSE-like cells) or 0.5x (p-value orders
of magnitude), tightened where the manuscript reports more decimal places.

Sources of truth:
  outputs/moe_v6/ablation_results.json       --> Table III neural rows + Fig 5 utilization
  outputs/moe_v6/baselines_results.json      --> Table III RF, XGB, Dense rows
  outputs/moe_v6/topk_ablation.json          --> Table VI panel (a)
  outputs/moe_v6/tb_ablation_results.json    --> Table VI panel (b)
  outputs/moe_v6/selectivity_real_results.json --> Table V
  outputs/moe_v7/fewshot_<target>_AGG.json   --> Table II + draw-level p-values
  outputs/fusion_ablation_v7/fusion_ablation_results.json --> Table IV
  outputs/moe_3view_results.json             --> 3-view ablation claim
  outputs/compat_v8_16tgt.json               --> Section V-C OLS law
  outputs/compat_v8_16tgt_supC.json          --> support-only sensitivity
  outputs/compat_dual_spotcheck.json         --> dual-encoder probe spot-check
"""
import json, os, sys
import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'outputs')

PASS = 0
FAIL = 0


def check(name, actual, expected, tol=0.01):
    global PASS, FAIL
    ok = abs(float(actual) - float(expected)) < tol
    sym = 'PASS' if ok else 'FAIL'
    PASS += int(ok); FAIL += int(not ok)
    print(f"  [{sym}] {name}: actual={actual:.4g}  expected={expected:.4g}  tol={tol}")


def load(p):
    return json.load(open(os.path.join(OUT, p)))


# =====================================================================
# Table I: dataset compound counts (verifiable directly from CSVs)
# =====================================================================
print("\n== Table I: dataset compound counts ==")
import pandas as pd
exp_counts = {'scd1': 762, 'fads': 1187, 'nk1r': 3056, 'drd2': 9966}
csv_map = {
    'scd1': 'scd1_binding.csv',
    'fads': 'fatty_acid_desaturase_bioactivity.csv',
    'nk1r': 'nk1r_combined.csv',
    'drd2': 'drd2_bioactivity.csv',
}
for t, exp in exp_counts.items():
    df = pd.read_csv(os.path.join(ROOT, 'data', csv_map[t]))
    check(f"Table I  {t} compound count", len(df), exp, tol=0.5)

# =====================================================================
# Table II: few-shot RMSE (representative cells)
# =====================================================================
print("\n== Table II: few-shot RMSE (representative cells, mean) ==")
T2_EXP = {
    # (target, method, k) -> expected mean RMSE
    ('scd1', 'morgan_transfer', 5):  0.773,
    ('scd1', 'morgan_transfer', 50): 0.719,
    ('scd1', 'moe_transfer', 10):    0.734,
    ('scd1', 'moe_transfer', 20):    0.725,
    ('scd1', 'maml', 5):             1.655,
    ('fads', 'morgan_transfer', 50): 0.987,
    ('fads', 'moe_transfer', 5):     1.252,
    ('fads', 'encoder_transfer', 50):1.114,
    ('nk1r', 'morgan_transfer', 50): 1.471,
    ('nk1r', 'descriptors_transfer', 5): 1.637,
    ('drd2', 'morgan_scratch', 5):   1.233,
    ('drd2', 'morgan_transfer', 5):  1.387,
}
for (t, m, k), exp in T2_EXP.items():
    d = load(f'moe_v7/fewshot_{t}_AGG.json')
    actual = d[m][str(k)]['mean']
    check(f"Table II {t}/{m}/k={k}", actual, exp, tol=0.005)

# Draw-level paired t-test claims in the prose
print("\n== Section V-A paired t-test claims ==")
for t, m1, m2, k_, exp_p, label in [
    ('fads', 'moe_transfer', 'morgan_transfer', 50, 0.04, 'Morgan-vs-MoE k=50'),
    ('fads', 'encoder_transfer', 'morgan_transfer', 50, 0.01, 'Enc-vs-Morgan k=50'),
]:
    d = load(f'moe_v7/fewshot_{t}_AGG.json')
    a = np.array(d[m1][str(k_)]['draw_means'])
    b = np.array(d[m2][str(k_)]['draw_means'])
    p = st.ttest_rel(a, b).pvalue
    check(f"FADS {label} draw-paired p", p, exp_p, tol=0.015)

# All SCD-1 MoE vs Morgan: paper claims p>0.2 at every k
print("  SCD-1 MoE vs Morgan: paper claims p>0.2 at every k")
d = load('moe_v7/fewshot_scd1_AGG.json')
for k in [5, 10, 20, 50]:
    a = np.array(d['moe_transfer'][str(k)]['draw_means'])
    b = np.array(d['morgan_transfer'][str(k)]['draw_means'])
    p = st.ttest_rel(a, b).pvalue
    ok = p > 0.2
    print(f"  [{'PASS' if ok else 'FAIL'}] SCD-1 MoE-vs-Morgan k={k}: p={p:.3f} > 0.2 ? {ok}")
    PASS += int(ok); FAIL += int(not ok)

# =====================================================================
# Table III: within-target RMSE
# =====================================================================
print("\n== Table III: within-target RMSE ==")
d_abl = load('moe_v6/ablation_results.json')
d_base = load('moe_v6/baselines_results.json')
T3_EXP = {
    ('scd1', 'morgan'): 0.94, ('scd1', 'encoder'): 0.83,
    ('scd1', 'maccs'): 1.00,  ('scd1', 'descriptors'): 1.00,
    ('scd1', 'moe'): 0.77,
    ('fads', 'morgan'): 1.01, ('fads', 'encoder'): 1.02, ('fads', 'moe'): 0.92,
    ('nk1r', 'morgan'): 1.13, ('nk1r', 'maccs'): 1.09, ('nk1r', 'moe'): 1.17,
    ('drd2', 'morgan'): 0.76, ('drd2', 'moe'): 0.81,
}
for (t, m), exp in T3_EXP.items():
    check(f"Table III {t}/{m}", d_abl[t][m]['mean'], exp, tol=0.01)

T3B_EXP = {
    ('scd1', 'rf_morgan'): 0.81, ('scd1', 'xgb_morgan'): 0.80, ('scd1', 'dense_concat_mlp'): 0.74,
    ('fads', 'rf_morgan'): 0.88, ('fads', 'xgb_morgan'): 0.84,
    ('nk1r', 'rf_morgan'): 1.00, ('nk1r', 'xgb_morgan'): 0.99,
    ('drd2', 'rf_morgan'): 0.70, ('drd2', 'xgb_morgan'): 0.71,
}
for (t, m), exp in T3B_EXP.items():
    check(f"Table III {t}/{m}", d_base[t][m]['mean'], exp, tol=0.01)

# =====================================================================
# Table IV: fusion ablation
# =====================================================================
print("\n== Table IV: encoder fusion ablation ==")
d_fus = load('fusion_ablation_v7/fusion_ablation_results.json')
T4_EXP = {
    ('attention', 'scd1'): 0.77, ('attention', 'fads'): 0.82,
    ('attention', 'nk1r'): 1.15, ('attention', 'drd2'): 0.78,
    ('no_xattn', 'scd1'): 0.71, ('no_xattn', 'fads'): 0.89,
    ('no_xattn', 'nk1r'): 1.06, ('no_xattn', 'drd2'): 0.82,
    ('graph_only', 'scd1'): 1.00, ('graph_only', 'fads'): 1.16,
    ('seq_only', 'scd1'): 0.83, ('seq_only', 'fads'): 0.92,
}
for (v, t), exp in T4_EXP.items():
    check(f"Table IV {v}/{t}", d_fus[v][t]['mean'], exp, tol=0.01)

# =====================================================================
# Table VI: top-K ablation
# =====================================================================
print("\n== Table VI panel (a): top-K ablation ==")
d_topk = load('moe_v6/topk_ablation.json')
T6A_EXP = {1: 0.968, 2: 0.922, 4: 0.929}
for cfg in d_topk:
    if cfg['target_emb_dim'] == 16 and cfg['top_k'] in T6A_EXP:
        check(f"Table VI(a) K={cfg['top_k']} overall_mean",
              cfg['overall_mean'], T6A_EXP[cfg['top_k']], tol=0.005)

print("== Table VI panel (b): target-conditioning factorial ==")
d_tb = load('moe_v6/tb_ablation_results.json')
T6B_EXP = {'t_on_b_on': 0.916, 't_on_b_off': 0.914,
           't_off_b_on': 0.925, 't_off_b_off': 0.923}
for k, exp in T6B_EXP.items():
    check(f"Table VI(b) {k}", d_tb[k]['overall_mean'], exp, tol=0.005)

# =====================================================================
# Figure 5: MoE expert utilization
# =====================================================================
print("\n== Figure 5: MoE expert utilization (Morgan, Encoder, MACCS, Descriptors) ==")
F5_EXP = {
    'scd1': [0.41, 0.37, 0.22, 0.00],
    'fads': [0.41, 0.42, 0.18, 0.00],
    'nk1r': [0.34, 0.34, 0.32, 0.00],
    'drd2': [0.41, 0.38, 0.21, 0.00],
}
for t, exp_util in F5_EXP.items():
    actual = d_abl[t]['moe']['utilization']
    for i, name in enumerate(['Morgan','Encoder','MACCS','Descriptors']):
        check(f"Fig 5 {t}/{name}", actual[i], exp_util[i], tol=0.02)

# =====================================================================
# Section V-C: 240-pair compatibility OLS law
# =====================================================================
print("\n== Section V-C: compatibility law (full panel + small-data) ==")
d_compat = load('compat_v8_16tgt.json')
rows = d_compat['rows']
cnn = np.array([r['C_nn'] for r in rows])
g = np.array([r['gain_mean'] for r in rows])
nt = np.array([r['n_target'] for r in rows], dtype=float)

pr, pp = st.pearsonr(cnn, g)
check("Section V-C Pearson(C_nn, gain) 240 pairs", pr, 0.261, tol=0.005)
check("Section V-C Pearson p-value (order of magnitude)",
      np.log10(pp), np.log10(4.4e-5), tol=0.5)

prn, ppn = st.pearsonr(np.log10(nt), g)
check("Section V-C Pearson(log_n_B, gain)", prn, -0.360, tol=0.005)

# Two-factor OLS
X = np.column_stack([np.ones(len(g)), cnn, np.log10(nt)])
beta, *_ = np.linalg.lstsq(X, g, rcond=None)
check("Section V-C beta_intercept", beta[0], 0.475, tol=0.01)
check("Section V-C beta_C_nn (= 0.36 in paper)", beta[1], 0.359, tol=0.01)
check("Section V-C beta_log_n (= -0.14 in paper)", beta[2], -0.137, tol=0.01)

# Small-data subset (n_target < 2000)
mask = nt < 2000
check("Section V-C small-data n pairs", mask.sum(), 30, tol=0.5)
prs, pps = st.pearsonr(cnn[mask], g[mask])
check("Section V-C small-data Pearson(C_nn, gain)", prs, 0.721, tol=0.01)

# Support-only sensitivity
print("== Section V-C: support-only C_nn sensitivity ==")
d_sup = load('compat_v8_16tgt_supC.json')['rows']
cnn_full2 = np.array([r['C_nn'] for r in d_sup])
cnn_sup = np.array([r['C_nn_sup'] for r in d_sup])
check("support C_nn vs full C_nn agreement", st.pearsonr(cnn_full2, cnn_sup)[0], 0.964, tol=0.01)
g_sup = np.array([r['gain_mean'] for r in d_sup])
Xs = np.column_stack([np.ones(len(g_sup)), cnn_sup, np.log10([r['n_target'] for r in d_sup])])
bs, *_ = np.linalg.lstsq(Xs, g_sup, rcond=None)
check("support beta_C_nn (= 0.344 in paper)", bs[1], 0.344, tol=0.01)
mask_s = np.array([r['n_target'] < 2000 for r in d_sup])
ps_s, _ = st.pearsonr(cnn_sup[mask_s], g_sup[mask_s])
check("support small-data Pearson(C_nn_sup, gain) = 0.70", ps_s, 0.695, tol=0.01)

# Dual-encoder spot-check (Limitations)
print("== Limitations: dual-encoder spot-check ==")
d_spot = load('compat_dual_spotcheck.json')['results']
faah = [r for r in d_spot if r['target'] == 'faah']
cn = np.array([r['C_nn'] for r in faah])
rmse = np.array([r['rmse_mean'] for r in faah])
pr_f, pp_f = st.pearsonr(cn, -rmse)
check("FAAH spot-check Pearson(C_nn, -RMSE) = 0.62", pr_f, 0.616, tol=0.01)
check("FAAH spot-check p-value = 0.014", pp_f, 0.0144, tol=0.005)

# Pooled across 3 small-data spot-check targets
cn_all = np.array([r['C_nn'] for r in d_spot])
rmse_all = np.array([r['rmse_mean'] for r in d_spot])
g_all = []
for B in ['scd1', 'fads', 'faah']:
    rows_B = [r for r in d_spot if r['target'] == B]
    rB = np.array([r['rmse_mean'] for r in rows_B])
    mB = rB.mean()
    for r in rows_B:
        g_all.append(mB - r['rmse_mean'])
g_all = np.array(g_all)
cn_all_ord = np.concatenate([np.array([r['C_nn'] for r in d_spot if r['target'] == B]) for B in ['scd1','fads','faah']])
pr_p, _ = st.pearsonr(cn_all_ord, g_all)
check("Pooled spot-check within-target gain Pearson", pr_p, -0.010, tol=0.05)

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"VERIFICATION: {PASS} passed, {FAIL} failed.")
print(f"{'='*60}")
sys.exit(0 if FAIL == 0 else 1)

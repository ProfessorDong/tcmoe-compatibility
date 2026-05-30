"""
Two reviewer-anticipation analyses:

(A) Support-only C_nn sensitivity.
    The headline compat law uses C_nn computed from the FULL B set
    (B_full -> A_train).  A reviewer can point out this is transductive
    (the test molecules influence source selection).  Here we recompute
    C_nn using only the SUPPORT compounds (k=20 scaffold support, the
    same draws used in the main study) and refit the OLS law.  If the
    relationship survives a support-only metric, the pre-screen claim is
    cleanly prospective.

(B) Mixed-effects refit with random intercepts for source and target.
    The 240 pairs are not independent (each source appears in 15 pairs,
    each target in 15).  We refit
        gain ~ C_nn + log10 n_target + (1|source) + (1|target)
    with statsmodels.MixedLM.  This produces dependence-aware SEs and
    p-values that complement the parametric OLS t-tests and the existing
    permutation test.
"""
import sys, json
import numpy as np
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from run_moe_predictor import load_target, scaffold_support_query


def morgan_bits(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)
    return [f for f in out if f is not None]


def compat_nn(A_fps, B_fps):
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(b, A_fps)) for b in B_fps]))


def ols(X, y):
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n, p = Xd.shape
    s2 = (resid @ resid) / (n - p)
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * st.t.sf(np.abs(t), n - p)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return beta, se, t, pv, r2


def main():
    rows_main = json.load(open('outputs/compat_v8_16tgt.json'))['rows']
    targets = sorted({r['target'] for r in rows_main} | {r['source'] for r in rows_main})

    print("Loading targets...", flush=True)
    data = {}
    for t in targets:
        smi, _ = load_target(t)
        tr, _, _ = __import__('run_moe_predictor').scaffold_split_indices(smi, seed=42)
        data[t] = {'smi': smi, 'tr_smi': [smi[i] for i in tr]}

    # --------------------------------------------------------------------
    # (A) Support-only C_nn at k=20, averaged over 5 draws (the same seeds
    #     used in compat_v8 so the protocol matches exactly).
    # --------------------------------------------------------------------
    print("\n=== (A) Support-only C_nn (k=20, 5 draws) ===")
    A_fps = {t: morgan_bits(data[t]['tr_smi']) for t in targets}
    print(f"  source fingerprints ready ({sum(len(v) for v in A_fps.values())} total)")

    cnn_support = {}
    for B in targets:
        per_draw = []
        for draw in range(5):
            sup, _ = scaffold_support_query(data[B]['smi'], 20, seed=42 + draw * 1000 + 20)
            B_smi_sup = [data[B]['smi'][i] for i in sup]
            B_fps_sup = morgan_bits(B_smi_sup)
            for A in targets:
                if A == B: continue
                c = compat_nn(A_fps[A], B_fps_sup)
                per_draw.append((A, B, draw, c))
        # aggregate
        for A in targets:
            if A == B: continue
            vals = [v for (a, b, d, v) in per_draw if a == A and b == B]
            cnn_support[(A, B)] = float(np.mean(vals))
        print(f"  computed support-only C_nn for held-out {B}", flush=True)

    # Build rows with both metrics
    new_rows = []
    for r in rows_main:
        rr = dict(r)
        rr['C_nn_sup'] = cnn_support[(r['source'], r['target'])]
        new_rows.append(rr)

    # Correlation: support C_nn vs full C_nn
    cn_full = np.array([r['C_nn'] for r in new_rows])
    cn_sup = np.array([r['C_nn_sup'] for r in new_rows])
    pr, pp = st.pearsonr(cn_full, cn_sup)
    print(f"  Pearson(full C_nn, support C_nn) over 240 pairs: r={pr:+.3f} (p={pp:.3g})")

    # Refit single-factor and two-factor OLS with support C_nn
    g = np.array([r['gain_mean'] for r in new_rows])
    nt = np.array([r['n_target'] for r in new_rows], dtype=float)

    pr_g, pp_g = st.pearsonr(cn_sup, g)
    sr_g, sp_g = st.spearmanr(cn_sup, g)
    print(f"  Pearson(C_nn_sup, gain) all 240 pairs: r={pr_g:+.3f} (p={pp_g:.3g})")
    print(f"  Spearman: rho={sr_g:+.3f} (p={sp_g:.3g})")

    print("\n=== Two-factor OLS with support-only C_nn ===")
    X = np.column_stack([cn_sup, np.log10(nt)])
    beta, se, t, pv, r2 = ols(X, g)
    print(f"  intercept   = {beta[0]:+.3f} (p={pv[0]:.4g})")
    print(f"  C_nn_sup    = {beta[1]:+.3f} (p={pv[1]:.4g})")
    print(f"  log10 n_tgt = {beta[2]:+.3f} (p={pv[2]:.4g})")
    print(f"  R^2 = {r2:.3f}")

    # Small-data subset
    mask = nt < 2000
    pr_s, pp_s = st.pearsonr(cn_sup[mask], g[mask])
    print(f"  small-data subset (n_target<2000, n={mask.sum()}): "
          f"Pearson(C_nn_sup, gain) r={pr_s:+.3f} p={pp_s:.3g}")

    # Save artifact
    json.dump({'rows': new_rows}, open('outputs/compat_v8_16tgt_supC.json', 'w'), indent=2)
    print(f"\n  saved outputs/compat_v8_16tgt_supC.json")

    # --------------------------------------------------------------------
    # (B) Mixed-effects refit with random intercepts for source and target.
    # --------------------------------------------------------------------
    print("\n=== (B) Mixed-effects refit (random intercepts for source AND target) ===")
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame([{
            'gain': r['gain_mean'],
            'cnn': r['C_nn'],
            'log_n': np.log10(r['n_target']),
            'source': r['source'],
            'target': r['target'],
        } for r in rows_main])

        # Crossed random effects via variance components
        md = smf.mixedlm("gain ~ cnn + log_n", df, groups=df['target'],
                         vc_formula={'source': '0 + C(source)'})
        mf = md.fit(reml=True, method='lbfgs', maxiter=200)
        print(mf.summary().tables[1])
        print("\n  random-effect variance:")
        print(f"    target group variance: {mf.cov_re.iloc[0,0]:.4f}")
        print(f"    source variance comp:  {mf.vcomp[0]:.4f}")
        print(f"    residual variance:     {mf.scale:.4f}")
    except Exception as e:
        print(f"  Mixed-effects failed: {type(e).__name__}: {e}")
        print("  Falling back to cluster-robust SEs by source and target (Cameron-Gelbach-Miller).")
        # Two-way cluster-robust SE via the union estimator:
        # Var_cluster_src + Var_cluster_tgt - Var_cluster_src_AND_tgt
        # For pair-level data with two-way clusters, this is the standard.
        import pandas as pd
        df = pd.DataFrame([{
            'gain': r['gain_mean'], 'cnn': r['C_nn'],
            'log_n': np.log10(r['n_target']),
            'source': r['source'], 'target': r['target'],
        } for r in rows_main])
        Xd = np.column_stack([np.ones(len(df)), df['cnn'], df['log_n']])
        y = df['gain'].values
        beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
        resid = y - Xd @ beta
        XtX_inv = np.linalg.inv(Xd.T @ Xd)

        def cluster_meat(groups):
            M = np.zeros((Xd.shape[1], Xd.shape[1]))
            for g in df[groups].unique():
                idx = (df[groups] == g).values
                u = (Xd[idx].T @ resid[idx]).reshape(-1, 1)
                M += u @ u.T
            return M

        Vs = XtX_inv @ cluster_meat('source') @ XtX_inv
        Vt = XtX_inv @ cluster_meat('target') @ XtX_inv
        # intersection: pairs where (source,target) match
        df['pair'] = df['source'] + '_' + df['target']
        Vst = XtX_inv @ cluster_meat('pair') @ XtX_inv
        V = Vs + Vt - Vst
        se = np.sqrt(np.maximum(np.diag(V), 0))
        z = beta / se
        pv = 2 * st.norm.sf(np.abs(z))
        for name, b, s, p in zip(['intercept', 'C_nn', 'log_n'], beta, se, pv):
            print(f"  {name:10s}: beta={b:+.3f}  cluster-SE={s:.4f}  z={b/s:+.2f}  p={p:.4g}")


if __name__ == '__main__':
    main()

"""
Items (1) and (2) from the reviewer-anticipation pass:

(1) Cross-family LOO validation of the compatibility-transfer law.
    For each pharmacology sub-family s, refit the two-factor OLS
        gain ~ C_nn + log10(n_target)
    on pairs whose source and target are BOTH outside s, then predict gain
    on held-out pairs (target in s, source outside s).  This tests whether
    the law generalizes to predict transfer to unseen sub-families rather
    than merely fitting in-sample.

(2) Permutation test on beta_1 (C_nn coefficient) to give a non-parametric
    p-value that does not depend on Gaussian / homoscedasticity assumptions
    of the t-test.
"""
import sys, json
import numpy as np
from scipy import stats as st

# Sub-family mapping for the 16 targets in the panel.
SUBFAM = {
    'scd1':    'lipid_enzyme',   'fads':    'lipid_enzyme',  'faah':   'lipid_enzyme',
    'ptgs2':   'other_enzyme',   'ache':    'other_enzyme',
    'drd2':    'aminergic_gpcr', 'drd3':    'aminergic_gpcr','htr2a':  'aminergic_gpcr',
    'oprm1':   'opioid_gpcr',    'oprk1':   'opioid_gpcr',
    'nk1r':    'peptide_gpcr',
    'cnr2':    'cannabinoid_gpcr',
    'adora2a': 'adenosine_gpcr',
    'egfr':    'kinase',         'kdr':     'kinase',
    'herg':    'ion_channel',
}


def ols(X, y):
    """OLS with intercept. Returns beta (3-vector: intercept, C_nn, log10 n)."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def predict(X, beta):
    Xd = np.column_stack([np.ones(len(X)), X])
    return Xd @ beta


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'outputs/compat_v8_16tgt.json'
    d = json.load(open(path))
    rows = d['rows']
    for r in rows:
        r['source_fam'] = SUBFAM[r['source']]
        r['target_fam'] = SUBFAM[r['target']]

    families = sorted(set(SUBFAM.values()))
    print(f"=== (1) Cross-family LOO validation ({len(families)} sub-families) ===")
    print("Train: pairs whose source AND target are both outside the held-out family.")
    print("Test:  pairs whose target is in the held-out family (source outside it).\n")

    all_pred, all_obs = [], []
    per_fold = []
    for fam in families:
        train = [r for r in rows
                 if r['source_fam'] != fam and r['target_fam'] != fam]
        test = [r for r in rows
                if r['target_fam'] == fam and r['source_fam'] != fam]
        if len(test) < 4:
            print(f"  hold out {fam:18s}  (test={len(test)})  skipped (too few test pairs)")
            continue
        X_tr = np.array([[r['C_nn'], np.log10(r['n_target'])] for r in train])
        y_tr = np.array([r['gain_mean'] for r in train])
        X_te = np.array([[r['C_nn'], np.log10(r['n_target'])] for r in test])
        y_te = np.array([r['gain_mean'] for r in test])
        beta = ols(X_tr, y_tr)
        y_pred = predict(X_te, beta)
        pr, pp = st.pearsonr(y_pred, y_te)
        per_fold.append({'fam': fam, 'n_test': len(test), 'n_train': len(train),
                         'beta': beta.tolist(), 'pearson_r': float(pr), 'pearson_p': float(pp)})
        print(f"  hold out {fam:18s}  train n={len(train):3d}  test n={len(test):3d}  "
              f"refit beta=[int {beta[0]:+.2f}, Cnn {beta[1]:+.2f}, logN {beta[2]:+.2f}]  "
              f"predicted vs observed r={pr:+.3f} (p={pp:.3g})")
        all_pred.extend(y_pred); all_obs.extend(y_te)

    all_pred, all_obs = np.array(all_pred), np.array(all_obs)
    pr, pp = st.pearsonr(all_pred, all_obs)
    rho, rp = st.spearmanr(all_pred, all_obs)
    print(f"\nPooled cross-family-CV held-out predictions vs observed (N={len(all_pred)} pairs):")
    print(f"  Pearson  r={pr:+.3f} (p={pp:.3g})")
    print(f"  Spearman rho={rho:+.3f} (p={rp:.3g})")
    # cross-validated R^2 (predictions are out-of-fold)
    ss_res = ((all_obs - all_pred) ** 2).sum()
    ss_tot = ((all_obs - all_obs.mean()) ** 2).sum()
    r2_cv = 1 - ss_res / ss_tot
    print(f"  cross-validated R^2 = {r2_cv:+.3f}")

    # stability of refit coefficients across folds
    betas = np.array([f['beta'] for f in per_fold])
    print(f"\nRefit coefficient stability across {len(per_fold)} folds:")
    print(f"  beta_intercept range: [{betas[:,0].min():+.3f}, {betas[:,0].max():+.3f}] "
          f"mean={betas[:,0].mean():+.3f}")
    print(f"  beta_C_nn      range: [{betas[:,1].min():+.3f}, {betas[:,1].max():+.3f}] "
          f"mean={betas[:,1].mean():+.3f}")
    print(f"  beta_logN      range: [{betas[:,2].min():+.3f}, {betas[:,2].max():+.3f}] "
          f"mean={betas[:,2].mean():+.3f}")

    # ====================================================================
    print(f"\n=== (2) Permutation test on beta_1 (C_nn coefficient) ===")
    X = np.array([[r['C_nn'], np.log10(r['n_target'])] for r in rows])
    y = np.array([r['gain_mean'] for r in rows])
    beta_obs = ols(X, y)
    print(f"observed beta_C_nn = {beta_obs[1]:+.4f}")

    rng = np.random.default_rng(42)
    n_perm = 5000
    perm_betas = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_betas[i] = ols(X, y_perm)[1]
    p_two = (np.abs(perm_betas) >= np.abs(beta_obs[1])).sum() / n_perm
    p_one = (perm_betas >= beta_obs[1]).sum() / n_perm
    print(f"two-sided empirical p ({n_perm} permutations) = {p_two:.4g}")
    print(f"one-sided empirical p ({n_perm} permutations) = {p_one:.4g}")
    print(f"max permuted |beta_1| = {np.abs(perm_betas).max():.4f} "
          f"(observed |beta_1| = {abs(beta_obs[1]):.4f})")

    # also for the small-data subset
    print(f"\n=== Permutation test on small-data subset (n_target < 2000) ===")
    mask = np.array([r['n_target'] < 2000 for r in rows])
    Xs = X[mask]; ys = y[mask]
    bs = ols(Xs, ys)
    pb = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(ys)
        pb[i] = ols(Xs, yp)[1]
    p_two_s = (np.abs(pb) >= np.abs(bs[1])).sum() / n_perm
    print(f"n pairs = {mask.sum()}, observed beta_C_nn = {bs[1]:+.4f}")
    print(f"two-sided empirical p ({n_perm} permutations) = {p_two_s:.4g}")


if __name__ == '__main__':
    main()

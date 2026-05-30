"""
Analyze the pairwise compatibility -> transfer-gain results.
Tests whether source--target chemical compatibility predicts few-shot transfer
gain, controlling for the held-out target's data scarcity. Produces:
  - single-factor correlations (gain vs C_nn, gain vs C_scaf)
  - two-factor OLS: gain ~ C_nn + log10(n_target)   (manual OLS with t-tests)
  - stratified: correlation within small-data targets
  - a scatter figure (gain vs C_nn, colored by target size) for the paper.
"""
import sys, json
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else 'outputs/compat_v8_10tgt.json'
FIGOUT = sys.argv[2] if len(sys.argv) > 2 else \
    '/home/dong/Workspace/WritePaper/theranostics/JournalPapers_prediction/figures/compat_gain.eps'
SMALL_THRESH = 2000  # "small-data" held-out target if n_target < this


def ols(X, y):
    """Manual OLS with intercept; returns coefs, se, t, p, R2."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n, p = Xd.shape
    dof = n - p
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * st.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return beta, se, t, pv, r2


def main():
    d = json.load(open(PATH))
    rows = d['rows']
    cnn = np.array([r['C_nn'] for r in rows])
    cscaf = np.array([r['C_scaf'] for r in rows])
    g = np.array([r['gain_mean'] for r in rows])
    ntar = np.array([r['n_target'] for r in rows], dtype=float)
    print(f"targets: {d['targets']}")
    print(f"N ordered pairs = {len(rows)}\n")

    print("=== single-factor correlations (all pairs) ===")
    for name, c in [('C_nn', cnn), ('C_scaf', cscaf)]:
        pr, pp = st.pearsonr(c, g); sr, sp = st.spearmanr(c, g)
        print(f"  {name}: Pearson r={pr:+.3f} (p={pp:.4g}) | Spearman={sr:+.3f} (p={sp:.4g})")
    pr, pp = st.pearsonr(np.log10(ntar), g)
    print(f"  log10(n_target): Pearson r={pr:+.3f} (p={pp:.4g})")

    print("\n=== two-factor OLS: gain ~ C_nn + log10(n_target) ===")
    X = np.column_stack([cnn, np.log10(ntar)])
    beta, se, t, pv, r2 = ols(X, g)
    print(f"  intercept   = {beta[0]:+.3f} (p={pv[0]:.4g})")
    print(f"  C_nn        = {beta[1]:+.3f} (p={pv[1]:.4g})")
    print(f"  log10 n_tgt = {beta[2]:+.3f} (p={pv[2]:.4g})")
    print(f"  R^2 = {r2:.3f}")

    print("\n=== stratified: small-data held-out targets (n_target < %d) ===" % SMALL_THRESH)
    mask = ntar < SMALL_THRESH
    print(f"  n small-data pairs = {mask.sum()}")
    if mask.sum() >= 4:
        pr, pp = st.pearsonr(cnn[mask], g[mask]); sr, sp = st.spearmanr(cnn[mask], g[mask])
        print(f"  C_nn vs gain (small-data targets): Pearson r={pr:+.3f} (p={pp:.4g}) | Spearman={sr:+.3f} (p={sp:.4g})")

    # partial: within-target Spearman (fix target, vary source), averaged
    print("\n=== within-target (fix target, vary source) Spearman, pooled ===")
    bytar = {}
    for r in rows:
        bytar.setdefault(r['target'], []).append((r['C_nn'], r['gain_mean']))
    rhos = []
    for tb, lst in bytar.items():
        if len(lst) >= 3:
            cc = [x[0] for x in lst]; gg = [x[1] for x in lst]
            rho, _ = st.spearmanr(cc, gg); rhos.append(rho)
    print(f"  mean within-target Spearman(C_nn,gain) = {np.nanmean(rhos):+.3f} over {len(rhos)} targets")

    # scatter figure
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sc = ax.scatter(cnn, g, c=np.log10(ntar), cmap='viridis', s=45,
                    edgecolor='black', linewidth=0.4)
    sl, ic, r, p, _ = st.linregress(cnn, g)
    xs = np.linspace(cnn.min(), cnn.max(), 50)
    p_label = f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
    ax.plot(xs, sl * xs + ic, 'r--', lw=1.5,
            label=f'fit (r={r:.2f}, p={p_label})')
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Source$\\rightarrow$target compatibility  $C_{\\mathrm{nn}}$', fontsize=12)
    ax.set_ylabel('Few-shot transfer gain (RMSE)', fontsize=12)
    cb = plt.colorbar(sc, ax=ax); cb.set_label('$\\log_{10}$ target size', fontsize=11)
    ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGOUT, format='eps', bbox_inches='tight')
    plt.savefig(FIGOUT.replace('.eps', '.png'), dpi=200, bbox_inches='tight')
    print(f"\nsaved figure {FIGOUT}")


if __name__ == '__main__':
    main()

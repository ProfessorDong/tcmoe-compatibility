"""
Item 1: Does source-target chemical COMPATIBILITY predict few-shot TRANSFER GAIN?

Pairwise single-source transfer: for every ordered pair (source A -> target B),
pretrain a Morgan-FP MLP (the strongest transfer method) on A's training split,
adapt to B with k support compounds, and measure
    gain(A->B,k) = RMSE_scratch(B,k) - RMSE_transfer(A->B,k)   (>0 means transfer helps).
Compatibility metrics C(A->B), computed from molecular structure only:
    nn   = mean over B of max Tanimoto to A's training molecules (2048-bit Morgan)
    scaf = fraction of B's Murcko scaffolds present in A's training set
We then test whether C predicts gain across all ordered pairs.

Targets are read from a manifest so the same code runs on the 4-target and the
expanded set. 5 pretraining draws x k in {5,10,20,50}.
"""
import sys, json, copy, os, argparse
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
sys.path.insert(0, '.')
from run_moe_predictor import (load_target, scaffold_split_indices,
    scaffold_support_query, make_baseline_mlp, murcko_scaffold, TARGET_PATHS)
from data.featurizer_views import batch_morgan
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
KS = [5, 10, 20, 50]
NDRAW = 5
SIM_CAP = 2000  # subsample cap for Tanimoto metric (mean-max NN is stable)


def morgan_bits(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)
    return [f for f in out if f is not None]


def compat_nn(A_fps, B_fps, seed=0):
    rng = np.random.default_rng(seed)
    A = A_fps if len(A_fps) <= SIM_CAP else [A_fps[i] for i in rng.choice(len(A_fps), SIM_CAP, replace=False)]
    B = B_fps if len(B_fps) <= SIM_CAP else [B_fps[i] for i in rng.choice(len(B_fps), SIM_CAP, replace=False)]
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(b, A)) for b in B]))


def compat_scaf(A_smiles, B_smiles):
    A = set(murcko_scaffold(s) for s in A_smiles)
    return float(np.mean([murcko_scaffold(s) in A for s in B_smiles]))


def mlp():
    return make_baseline_mlp(1024).to(DEV)


def fit(model, X, y, epochs, lr=5e-4):
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X, device=DEV); yt = torch.tensor(y, device=DEV)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = F.mse_loss(model(Xt).squeeze(-1), yt); loss.backward(); opt.step()
    return model


def pretrain(Xtr, ytr, Xval, yval, epochs=200, patience=20, lr=5e-4):
    m = mlp(); opt = Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, device=DEV)
    Xv = torch.tensor(Xval, device=DEV); yv = torch.tensor(yval, device=DEV)
    bs = 64; n = len(yt); idx = np.arange(n)
    best = copy.deepcopy(m.state_dict()); bv = float('inf'); since = 0
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for s in range(0, n, bs):
            j = idx[s:s+bs]
            opt.zero_grad(); loss = F.mse_loss(m(Xt[j]).squeeze(-1), yt[j]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(F.mse_loss(m(Xv).squeeze(-1), yv).sqrt().item())
        if v < bv - 1e-4: bv = v; best = copy.deepcopy(m.state_dict()); since = 0
        else:
            since += 1
            if since >= patience: break
    m.load_state_dict(best); return m


def rmse(model, X, y):
    model.eval()
    with torch.no_grad():
        return float(F.mse_loss(model(torch.tensor(X, device=DEV)).squeeze(-1),
                                torch.tensor(y, device=DEV)).sqrt().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2'])
    ap.add_argument('--out', default='outputs/compat_v8.json')
    ap.add_argument('--ft_epochs', type=int, default=100)
    args = ap.parse_args()
    T = args.targets

    data = {}
    for t in T:
        smi, y = load_target(t)
        X = batch_morgan(smi).astype(np.float32); y = y.astype(np.float32)
        tr, va, te = scaffold_split_indices(smi, seed=42)
        data[t] = {'smi': smi, 'X': X, 'y': y, 'tr': tr, 'va': va, 'te': te,
                   'fps_tr': morgan_bits([smi[i] for i in tr]),
                   'fps_full': morgan_bits(smi),
                   'smi_tr': [smi[i] for i in tr]}
        print(f"loaded {t}: n={len(smi)} (tr={len(tr)})", flush=True)

    # compatibility (structure only) for all ordered pairs
    compat = {}
    for a in T:
        for b in T:
            if a == b: continue
            cnn = compat_nn(data[a]['fps_tr'], data[b]['fps_full'])
            csc = compat_scaf(data[a]['smi_tr'], data[b]['smi'])
            compat[(a, b)] = {'nn': cnn, 'scaf': csc}
            print(f"  C({a}->{b}): nn={cnn:.3f} scaf={csc:.3f}", flush=True)

    # precompute scratch RMSE per (B,k,draw) (independent of source A)
    scratch = {b: {k: [] for k in KS} for b in T}
    for b in T:
        d = data[b]
        for draw in range(NDRAW):
            for k in KS:
                np.random.seed(1000 + draw); torch.manual_seed(1000 + draw)
                sup, qry = scaffold_support_query(d['smi'], k, seed=42 + draw * 1000 + k)
                m = fit(mlp(), d['X'][sup], d['y'][sup], args.ft_epochs)
                scratch[b][k].append(rmse(m, d['X'][qry], d['y'][qry]))

    # pairwise transfer: pretrain on A per draw, adapt to every B
    pair_gain = {(a, b): {k: [] for k in KS} for a in T for b in T if a != b}
    for draw in range(NDRAW):
        for a in T:
            np.random.seed(2000 + draw); torch.manual_seed(2000 + draw)
            da = data[a]
            base = pretrain(da['X'][da['tr']], da['y'][da['tr']], da['X'][da['va']], da['y'][da['va']])
            for b in T:
                if a == b: continue
                db = data[b]
                for k in KS:
                    sup, qry = scaffold_support_query(db['smi'], k, seed=42 + draw * 1000 + k)
                    m = fit(copy.deepcopy(base), db['X'][sup], db['y'][sup], args.ft_epochs)
                    tr_rmse = rmse(m, db['X'][qry], db['y'][qry])
                    gain = scratch[b][k][draw] - tr_rmse
                    pair_gain[(a, b)][k].append(gain)
            print(f"  draw {draw} source {a} done", flush=True)

    # aggregate
    rows = []
    for a in T:
        for b in T:
            if a == b: continue
            g_all = [g for k in KS for g in pair_gain[(a, b)][k]]
            row = {'source': a, 'target': b,
                   'n_source': len(data[a]['smi']), 'n_target': len(data[b]['smi']),
                   'C_nn': compat[(a, b)]['nn'], 'C_scaf': compat[(a, b)]['scaf'],
                   'gain_mean': float(np.mean(g_all)), 'gain_std': float(np.std(g_all))}
            for k in KS:
                row[f'gain_k{k}'] = float(np.mean(pair_gain[(a, b)][k]))
            rows.append(row)
            print(f"  {a}->{b}: C_nn={row['C_nn']:.3f} gain={row['gain_mean']:+.3f}", flush=True)
    json.dump({'targets': T, 'rows': rows}, open(args.out, 'w'), indent=2)
    print("saved", args.out)


if __name__ == '__main__':
    main()

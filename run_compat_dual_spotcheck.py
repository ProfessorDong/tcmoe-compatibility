"""
Dual-encoder spot-check of the compatibility-transfer law.

Validates that the predictive relationship between source--target chemical
compatibility C_nn and few-shot transfer gain is not specific to the
Morgan-FP MLP transfer probe used throughout the main compatibility study
(Section V-C of the manuscript).

Protocol:
  For each held-out small-data target B in {scd1, fads, faah} (the regime
  where the Morgan-FP law was strongest: Pearson r=0.72), vary the source A
  across the other 15 targets, pretrain a fresh dual GAT/BiGRU/cross-attention
  encoder + a regression head on A's training split (200 batches, fixed
  seed), then few-shot adapt to B at k=20 scaffold support across 5
  independent support/query resamplings drawn from the same seeds used in
  outputs/compat_v8_16tgt.json (cross-architecture comparability).

Per-pair RMSE is averaged over the 5 resamplings; within each B we then test
  (a) Pearson and Spearman between C_nn(A->B) and -RMSE_dual(A->B), and
  (b) Pearson between C_nn and the within-target gain
      g_B(A) = mean_A' RMSE_dual(A'->B) - RMSE_dual(A->B),
  which removes B's intrinsic difficulty so cross-family aggregation is
  meaningful.

A positive correlation in any of these tests replicates the Morgan-FP law
with a graph+sequence transfer probe.
"""
import sys, json, copy, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from run_moe_predictor import (load_target, scaffold_split_indices,
    scaffold_support_query, _GraphBatch)
from models.dual_encoder import DualEncoder
from data.dataset import DualDataset, collate_dual
from scipy import stats as st

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K = 20                     # support size; matches the regime where law is strongest
NDRAW = 5
HELD_OUT = ['scd1', 'fads', 'faah']
TARGETS_ALL = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3', 'htr2a', 'oprm1',
               'oprk1', 'cnr2', 'adora2a', 'faah', 'ptgs2', 'ache',
               'egfr', 'kdr', 'herg']
PRETRAIN_EPOCHS = 20
FT_EPOCHS = 50
LR_PRETRAIN = 1e-3
LR_FT = 5e-4
WD = 1e-4
OUT_PATH = 'outputs/compat_dual_spotcheck.json'
COMPAT_PATH = 'outputs/compat_v8_16tgt.json'


def build_model():
    enc = DualEncoder(
        node_features=75, edge_features=6,
        graph_hidden_dims=[128, 256, 128], vocab_size=128,
        smiles_hidden_dim=256,
        output_dim=128, num_heads=4, dropout=0.1,
        fusion_type='attention',
    ).to(DEV)
    head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                          nn.Dropout(0.2), nn.Linear(64, 1)).to(DEV)
    return enc, head


def pretrain_source(smi, y, epochs=PRETRAIN_EPOCHS, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, head = build_model()
    ds = DualDataset(smi, y.astype(np.float32), max_length=120)
    loader = DataLoader(ds, batch_size=64, shuffle=True,
                        collate_fn=collate_dual, drop_last=True)
    opt = Adam(list(enc.parameters()) + list(head.parameters()),
               lr=LR_PRETRAIN, weight_decay=WD)
    enc.train(); head.train()
    for ep in range(epochs):
        for b in loader:
            gb = _GraphBatch(b['graph']).to(DEV)
            tok = b['tokens'].to(DEV); ln = b['lengths']
            yy = b['labels'].to(DEV)
            opt.zero_grad()
            emb = enc(gb, tok, ln)
            pred = head(emb).squeeze(-1)
            loss = F.mse_loss(pred, yy)
            loss.backward(); opt.step()
    return {k: v.cpu() for k, v in enc.state_dict().items()}, \
           {k: v.cpu() for k, v in head.state_dict().items()}


def finetune_and_eval(enc_sd, head_sd, smi_sup, y_sup, smi_qry, y_qry,
                     epochs=FT_EPOCHS, lr=LR_FT, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, head = build_model()
    enc.load_state_dict({k: v.to(DEV) for k, v in enc_sd.items()})
    head.load_state_dict({k: v.to(DEV) for k, v in head_sd.items()})
    opt = Adam(list(enc.parameters()) + list(head.parameters()),
               lr=lr, weight_decay=WD)
    ds_sup = DualDataset(smi_sup, y_sup.astype(np.float32), max_length=120)
    bs = min(8, len(smi_sup))
    loader = DataLoader(ds_sup, batch_size=bs, shuffle=True,
                        collate_fn=collate_dual)
    enc.train(); head.train()
    for ep in range(epochs):
        for b in loader:
            gb = _GraphBatch(b['graph']).to(DEV)
            tok = b['tokens'].to(DEV); ln = b['lengths']
            yy = b['labels'].to(DEV)
            opt.zero_grad()
            emb = enc(gb, tok, ln)
            pred = head(emb).squeeze(-1)
            loss = F.mse_loss(pred, yy)
            loss.backward(); opt.step()
    enc.eval(); head.eval()
    ds_qry = DualDataset(smi_qry, np.zeros(len(smi_qry), dtype=np.float32),
                         max_length=120)
    loader_q = DataLoader(ds_qry, batch_size=128, shuffle=False,
                          collate_fn=collate_dual)
    preds = []
    with torch.no_grad():
        for b in loader_q:
            gb = _GraphBatch(b['graph']).to(DEV)
            tok = b['tokens'].to(DEV); ln = b['lengths']
            emb = enc(gb, tok, ln)
            preds.append(head(emb).squeeze(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return float(np.sqrt(np.mean((preds - y_qry) ** 2)))


def main():
    t_start = time.time()
    data = {}
    for t in TARGETS_ALL:
        smi, y = load_target(t)
        tr, va, te = scaffold_split_indices(smi, seed=42)
        data[t] = {'smi': smi, 'y': y, 'tr': tr,
                   'smi_tr': [smi[i] for i in tr], 'y_tr': y[tr]}
        print(f"loaded {t}: n={len(smi)} (tr={len(tr)})", flush=True)

    compat = json.load(open(COMPAT_PATH))
    C_nn = {(r['source'], r['target']): r['C_nn'] for r in compat['rows']}

    print(f"\n=== Pretraining encoders on {len(TARGETS_ALL)} sources ===", flush=True)
    pretrained = {}
    for A in TARGETS_ALL:
        t0 = time.time()
        enc_sd, head_sd = pretrain_source(data[A]['smi_tr'], data[A]['y_tr'])
        pretrained[A] = (enc_sd, head_sd)
        print(f"  {A:8s} pretrained in {time.time()-t0:.0f}s", flush=True)

    # Save intermediate state in case of interruption
    print(f"\n=== Fine-tuning: {len(HELD_OUT)} held-out x {len(TARGETS_ALL)-1} sources x {NDRAW} draws at k={K} ===", flush=True)
    results = []
    for B in HELD_OUT:
        for A in TARGETS_ALL:
            if A == B: continue
            rmses = []
            for draw in range(NDRAW):
                sup, qry = scaffold_support_query(
                    data[B]['smi'], K, seed=42 + draw * 1000 + K)
                smi_sup = [data[B]['smi'][i] for i in sup]
                y_sup = data[B]['y'][sup]
                smi_qry = [data[B]['smi'][i] for i in qry]
                y_qry = data[B]['y'][qry]
                rmse_val = finetune_and_eval(
                    pretrained[A][0], pretrained[A][1],
                    smi_sup, y_sup, smi_qry, y_qry, seed=draw)
                rmses.append(rmse_val)
            mean_rmse = float(np.mean(rmses))
            results.append({'source': A, 'target': B, 'k': K,
                            'rmse_mean': mean_rmse,
                            'rmse_std': float(np.std(rmses)),
                            'C_nn': C_nn[(A, B)]})
            print(f"  {A:8s} -> {B:6s}  RMSE={mean_rmse:.3f}+/-{np.std(rmses):.3f}  "
                  f"C_nn={C_nn[(A,B)]:.3f}", flush=True)

    json.dump({'targets_held_out': HELD_OUT, 'targets_all': TARGETS_ALL,
               'k': K, 'ndraws': NDRAW, 'results': results},
              open(OUT_PATH, 'w'), indent=2)
    print(f"\nsaved {OUT_PATH}", flush=True)

    print("\n=== Per-target compatibility correlations ===")
    pooled_C, pooled_negRMSE, pooled_gain = [], [], []
    per_B = {}
    for B in HELD_OUT:
        rows = [r for r in results if r['target'] == B]
        cn = np.array([r['C_nn'] for r in rows])
        rmse = np.array([r['rmse_mean'] for r in rows])
        mean_B = rmse.mean()
        gain = mean_B - rmse
        pr, pp = st.pearsonr(cn, -rmse); sr, sp = st.spearmanr(cn, -rmse)
        prg, ppg = st.pearsonr(cn, gain)
        per_B[B] = {'pearson_negRMSE': (float(pr), float(pp)),
                    'spearman_negRMSE': (float(sr), float(sp)),
                    'pearson_gain': (float(prg), float(ppg))}
        print(f"  {B} (n={len(rows)}): Pearson(C_nn,-RMSE) r={pr:+.3f} p={pp:.3g} | "
              f"Spearman {sr:+.3f} p={sp:.3g} | Pearson(C_nn,within-tgt gain) r={prg:+.3f} p={ppg:.3g}")
        pooled_C.extend(cn); pooled_negRMSE.extend(-rmse); pooled_gain.extend(gain)

    print("\n=== Pooled across 3 small-data held-out targets ===")
    pr, pp = st.pearsonr(pooled_C, pooled_negRMSE)
    sr, sp = st.spearmanr(pooled_C, pooled_negRMSE)
    print(f"  Pearson(C_nn, -RMSE) over N={len(pooled_C)} pairs: r={pr:+.3f} p={pp:.3g}")
    print(f"  Spearman(C_nn, -RMSE):                              rho={sr:+.3f} p={sp:.3g}")
    prg, ppg = st.pearsonr(pooled_C, pooled_gain)
    srg, spg = st.spearmanr(pooled_C, pooled_gain)
    print(f"  Pearson(C_nn, within-target gain):                  r={prg:+.3f} p={ppg:.3g}")
    print(f"  Spearman(C_nn, within-target gain):                 rho={srg:+.3f} p={spg:.3g}")

    print(f"\nTOTAL wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()

"""
3-view MoE check: does dropping the (near-unused) RDKit descriptor expert
change within-target accuracy? Trains a 3-view MoE (Morgan, encoder, MACCS)
jointly on all four targets, reusing the moe_v6 leak-free encoder so the ONLY
difference from the 4-view MoE is the removed descriptor expert. 5 seeds,
fixed seed-42 scaffold split (matches Table III).
"""
import sys, json, copy
import numpy as np, torch
import torch.nn.functional as F
from torch.optim import Adam
sys.path.insert(0, '.')
from run_moe_predictor import (ALL_TARGETS, TARGET_IDX, load_target,
    scaffold_split_indices, get_dual_encoder_embeddings, compute_views)
from models.moe_predictor import MoEPredictor

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ENC = 'outputs/moe_v6/dual_encoder_clean.pt'
V3 = ['morgan', 'encoder', 'maccs']  # descriptors dropped


def build():
    return MoEPredictor(view_dims=[1024, 128, 167], view_names=V3, n_targets=4,
                        top_k=2, target_emb_dim=16, load_balance=0.01, dropout=0.2).to(DEV)


def train(trv, trt, try_, vav, vat, vay, n_epochs=120, lr=5e-4, bs=128, wd=1e-4, patience=20):
    moe = build(); opt = Adam(moe.parameters(), lr=lr, weight_decay=wd)
    packs = [torch.tensor(trv[k]) for k in V3]
    tgt = torch.tensor(trt, dtype=torch.long); yy = torch.tensor(try_, dtype=torch.float32)
    n = packs[0].size(0); idxs = np.arange(n)
    vat_ = [torch.tensor(vav[k], device=DEV) for k in V3]
    vtg = torch.tensor(vat, device=DEV, dtype=torch.long)
    vyt = torch.tensor(vay, device=DEV, dtype=torch.float32)
    best = copy.deepcopy(moe.state_dict()); bv = float('inf'); since = 0
    for ep in range(n_epochs):
        moe.train(); np.random.shuffle(idxs)
        for s in range(0, n, bs):
            i = idxs[s:s + bs]
            xb = [v[i].to(DEV) for v in packs]; tb = tgt[i].to(DEV); yb = yy[i].to(DEV)
            opt.zero_grad(); pred, aux = moe(xb, tb, return_aux=True)
            (F.mse_loss(pred, yb) + aux['load_balance_loss']).backward(); opt.step()
        moe.eval()
        with torch.no_grad():
            v = float(F.mse_loss(moe(vat_, vtg), vyt).sqrt().item())
        if v < bv - 1e-4:
            bv = v; best = copy.deepcopy(moe.state_dict()); since = 0
        else:
            since += 1
            if since >= patience: break
    moe.load_state_dict(best); return moe


def ev(moe, v, t, y):
    moe.eval()
    xb = [torch.tensor(v[k], device=DEV) for k in V3]
    tb = torch.tensor(t, device=DEV, dtype=torch.long)
    with torch.no_grad():
        return float(F.mse_loss(moe(xb, tb), torch.tensor(y, device=DEV, dtype=torch.float32)).sqrt().item())


def main():
    data = {}
    for t in ALL_TARGETS:
        smi, y = load_target(t)
        emb = get_dual_encoder_embeddings(smi, DEV, weights_path=ENC)
        v = compute_views(smi, encoder_emb=emb)
        tr, va, te = scaffold_split_indices(smi, seed=42)
        data[t] = {'v': v, 'y': y.astype(np.float32), 'tr': tr, 'va': va, 'te': te}
    res = {t: [] for t in ALL_TARGETS}
    util = {t: [] for t in ALL_TARGETS}
    for seed in range(5):
        np.random.seed(42 + seed); torch.manual_seed(42 + seed)
        trv = {k: [] for k in V3}; vav = {k: [] for k in V3}
        trt = []; try_ = []; vat = []; vay = []
        for t in ALL_TARGETS:
            d = data[t]
            for k in V3:
                trv[k].append(d['v'][k][d['tr']]); vav[k].append(d['v'][k][d['va']])
            trt.append(np.full(len(d['tr']), TARGET_IDX[t], dtype=np.int64))
            vat.append(np.full(len(d['va']), TARGET_IDX[t], dtype=np.int64))
            try_.append(d['y'][d['tr']]); vay.append(d['y'][d['va']])
        trv = {k: np.concatenate(v) for k, v in trv.items()}
        vav = {k: np.concatenate(v) for k, v in vav.items()}
        trt = np.concatenate(trt); try_ = np.concatenate(try_)
        vat = np.concatenate(vat); vay = np.concatenate(vay)
        moe = train(trv, trt, try_, vav, vat, vay)
        for t in ALL_TARGETS:
            d = data[t]
            tev = {k: d['v'][k][d['te']] for k in V3}
            tetg = np.full(len(d['te']), TARGET_IDX[t], dtype=np.int64)
            res[t].append(ev(moe, tev, tetg, d['y'][d['te']]))
            xb = [torch.tensor(tev[k], device=DEV) for k in V3]
            util[t].append(moe.expert_utilization(xb, torch.tensor(tetg, device=DEV)).tolist())
        print(f"seed {seed}: " + "  ".join(f"{t}={res[t][-1]:.3f}" for t in ALL_TARGETS), flush=True)
    out = {}
    print("\n=== 3-view MoE (Morgan, encoder, MACCS) within-target RMSE ===")
    for t in ALL_TARGETS:
        m = float(np.mean(res[t])); s = float(np.std(res[t]))
        u = np.mean(util[t], axis=0).tolist()
        out[t] = {'mean': m, 'std': s, 'rmses': res[t], 'utilization_MEK': u}
        print(f"  {t}: {m:.3f}±{s:.3f}  util(M/E/K)={u[0]:.2f}/{u[1]:.2f}/{u[2]:.2f}")
    json.dump(out, open('outputs/moe_3view_results.json', 'w'), indent=2)
    print("saved outputs/moe_3view_results.json")


if __name__ == '__main__':
    main()

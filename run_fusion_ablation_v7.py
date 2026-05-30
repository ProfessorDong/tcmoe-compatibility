"""
Cross-attention fusion ablation (tests contribution ii).

For each fusion strategy, train a leak-free dual encoder on the union of the
four targets' TRAINING-split scaffolds, then evaluate the resulting encoder
VIEW by a single-view MLP (the 'encoder' row of Table III protocol): per-target
scaffold-split test RMSE, 5 model-init seeds, fixed seed-42 split.

Variants:
  attention  -- pre-pooling cross-attention + gated fusion (the paper's default)
  gate       -- pool each modality, gated fusion of the two pooled vectors (no x-attn)
  concat     -- pool each modality, concat + MLP (no x-attn)
  graph_only -- GAT pooled embedding only
  seq_only   -- BiGRU pooled embedding only

If pre-pooling cross-attention is the right design, 'attention' should give the
lowest encoder-view RMSE, especially on the small-data targets.
"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, '.')
from run_moe_predictor import (
    ALL_TARGETS, load_target, scaffold_split_indices,
    get_dual_encoder_embeddings, train_single_view, eval_single_view,
)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
OUT = 'outputs/fusion_ablation_v7'
os.makedirs(OUT, exist_ok=True)
SEED = 42
FUSIONS = ['attention', 'no_xattn', 'graph_only', 'seq_only']


def main():
    # per-target data + fixed seed-42 scaffold split
    data = {}
    for t in ALL_TARGETS:
        smi, y = load_target(t)
        tr, va, te = scaffold_split_indices(smi, seed=SEED)
        data[t] = {'smi': smi, 'y': y.astype(np.float32), 'tr': tr, 'va': va, 'te': te}

    # leak-free encoder training set = union of TRAINING-split scaffolds (all 4 targets)
    train_smiles, train_y = [], []
    for t in ALL_TARGETS:
        d = data[t]
        for i in d['tr']:
            train_smiles.append(d['smi'][i]); train_y.append(float(d['y'][i]))
    print(f'encoder training set: {len(train_smiles)} train-split molecules (leak-free)', flush=True)

    results = {}
    for ft in FUSIONS:
        print(f'\n######## fusion_type = {ft} ########', flush=True)
        wpath = f'{OUT}/encoder_{ft}.pt'
        if os.path.exists(wpath):
            os.remove(wpath)  # force fresh train for this variant
        emb = {}
        for t in ALL_TARGETS:
            emb[t] = get_dual_encoder_embeddings(
                data[t]['smi'], DEVICE, weights_path=wpath,
                train_smiles=train_smiles, train_y=train_y, fusion_type=ft
            ).astype(np.float32)
        results[ft] = {}
        for t in ALL_TARGETS:
            d = data[t]; E = emb[t]; y = d['y']
            rmses = []
            for s in range(5):
                np.random.seed(SEED + s); torch.manual_seed(SEED + s)
                model, _ = train_single_view(
                    E[d['tr']], y[d['tr']], E[d['va']], y[d['va']], DEVICE)
                rmses.append(eval_single_view(model, E[d['te']], y[d['te']], DEVICE))
            results[ft][t] = {'mean': float(np.mean(rmses)),
                              'std': float(np.std(rmses)), 'rmses': rmses}
            print(f'  {ft:10s} {t:5s}  encoder-view RMSE = {np.mean(rmses):.3f}'
                  f'±{np.std(rmses):.3f}', flush=True)
    json.dump(results, open(f'{OUT}/fusion_ablation_results.json', 'w'), indent=2)
    print('\nsaved', f'{OUT}/fusion_ablation_results.json')


if __name__ == '__main__':
    main()

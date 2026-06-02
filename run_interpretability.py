#!/usr/bin/env python
"""
SCD-1 interpretability stack for the prediction paper (Section V-D).
IG attribution on the Morgan view + a depth-4 decision-tree surrogate, over the
REAL SCD-1 dataset (load_target('scd1')) scored by the
SCD-1 Morgan-FP predictor, rather than molecules from the (dropped) generator.
Writes feature_attributions.json and extracted_rules.json.
Deterministic: fixed seeds, full real dataset.
"""
import os, sys, json
import numpy as np, pandas as pd, torch
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE); sys.path.insert(0, BASE)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, DataStructs
from models.property_predictor import PropertyPredictor
from run_moe_predictor import load_target

OUT = os.path.join(BASE, 'outputs', 'interpretability')

def load_predictor():
    ck = torch.load(os.path.join(BASE, 'outputs', 'scd1', 'predictor_scd1.pt'), map_location='cpu')
    p = PropertyPredictor(input_dim=1024, task_configs=ck['task_configs'], hidden_dims=[512,256], dropout=0.2)
    p.load_state_dict(ck['model_state_dict']); p.eval(); return p

def morgan_fp_with_info(smi, radius=2, nbits=1024):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None, None
    bi = {}; fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits, bitInfo=bi)
    a = np.zeros(nbits, dtype=np.float32); DataStructs.ConvertToNumpyArray(fp, a); return a, bi

def compute_descriptors(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        return {'MolWt':Descriptors.MolWt(m),'LogP':Crippen.MolLogP(m),'NumHDonors':Descriptors.NumHDonors(m),
                'NumHAcceptors':Descriptors.NumHAcceptors(m),'TPSA':Descriptors.TPSA(m),
                'NumRotatableBonds':Descriptors.NumRotatableBonds(m),'RingCount':Descriptors.RingCount(m),
                'NumAromaticRings':Descriptors.NumAromaticRings(m),'FractionCSP3':Descriptors.FractionCSP3(m),
                'NumHeavyAtoms':m.GetNumHeavyAtoms(),'QED':QED.qed(m)}
    except Exception: return None

# --- real SCD-1 molecules scored by the SCD-1 Morgan-FP predictor ---
smiles, _ = load_target('scd1')
pred = load_predictor()
preds = []
with torch.no_grad():
    for s in smiles:
        a,_ = morgan_fp_with_info(s)
        if a is None: preds.append(np.nan); continue
        preds.append(float(pred(torch.tensor(a).unsqueeze(0), task_name='pIC50')['pIC50'].item()))
df = pd.DataFrame({'smiles': smiles, 'pred_pIC50': preds}).dropna()
df_sorted = df.sort_values('pred_pIC50', ascending=False).reset_index(drop=True)
print(f"real SCD-1 molecules scored: {len(df_sorted)} | pred>=7: {(df_sorted['pred_pIC50']>=7).sum()}")

# ===== TASK 1: Integrated Gradients on Morgan view (top 100 by pred) =====
top100 = df_sorted.head(100)
n_steps = 50; all_attr = np.zeros(1024); bit_info_all = defaultdict(list); n_valid = 0
for _, row in top100.iterrows():
    fp_arr, bi = morgan_fp_with_info(row['smiles'])
    if fp_arr is None: continue
    for b, infos in bi.items():
        for (ca, rad) in infos: bit_info_all[b].append((row['smiles'], ca, rad))
    baseline = torch.zeros(1,1024); inp = torch.tensor(fp_arr).unsqueeze(0); attr = torch.zeros(1024, dtype=torch.float64)
    for step in range(n_steps+1):
        interp = (baseline + (step/n_steps)*(inp-baseline)).clone().requires_grad_(True)
        p = pred(interp, task_name='pIC50')['pIC50']; p.backward()
        if interp.grad is not None: attr += interp.grad.squeeze().double(); interp.grad=None
    attr = (inp.squeeze().double()-baseline.squeeze().double())*attr/(n_steps+1)
    all_attr += attr.detach().numpy(); n_valid += 1
avg = all_attr/max(n_valid,1); top_bits = np.argsort(np.abs(avg))[::-1][:20]
bit_sub = {}
for b in top_bits:
    b=int(b); s=set()
    for (smi,ca,rad) in bit_info_all.get(b,[])[:10]:
        m=Chem.MolFromSmiles(smi)
        if m is None: continue
        if rad==0: s.add(f"[{m.GetAtomWithIdx(ca).GetSymbol()}]")
        else:
            env=Chem.FindAtomEnvironmentOfRadiusN(m,rad,ca)
            if env:
                try:
                    sma=Chem.MolToSmarts(Chem.PathToSubmol(m,env))
                    if sma: s.add(sma)
                except Exception: pass
    bit_sub[str(b)] = list(s)[:5]
task1 = {"top_bits":[int(b) for b in top_bits],"bit_importance":[float(avg[b]) for b in top_bits],
         "bit_substructures":bit_sub,"mean_attribution_magnitude":float(np.mean(np.abs(avg))),"n_molecules":n_valid}
json.dump(task1, open(os.path.join(OUT,'feature_attributions.json'),'w'), indent=2)
print("\n=== TASK 1 (IG) top bits ===")
for b,imp in zip(task1['top_bits'][:8], task1['bit_importance'][:8]):
    print(f"  bit {b}: {imp:+.4f}  {bit_sub[str(b)]}")

# ===== TASK 2: decision-tree surrogate over the predictor's calls =====
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
hits = df_sorted[df_sorted['pred_pIC50']>=7.0].head(100)
non = df_sorted[df_sorted['pred_pIC50']<7.0].sample(n=min(100,len(df_sorted[df_sorted['pred_pIC50']<7.0])), random_state=42)
comb = pd.concat([hits,non], ignore_index=True)
names=['MolWt','LogP','NumHDonors','NumHAcceptors','TPSA','NumRotatableBonds','RingCount','NumAromaticRings','FractionCSP3','NumHeavyAtoms','QED']
X=[]; y=[]
for _,r in comb.iterrows():
    d=compute_descriptors(r['smiles'])
    if d is None: continue
    f=[d[k] for k in names]
    if any(v is None for v in f): continue
    X.append(f); y.append(1 if r['pred_pIC50']>=7.0 else 0)
X=np.array(X,dtype=np.float32); y=np.array(y)
dt=DecisionTreeClassifier(max_depth=4, random_state=42); dt.fit(X,y)
acc=float(accuracy_score(y, dt.predict(X)))
fi={n:float(v) for n,v in zip(names, dt.feature_importances_)}
task2={"accuracy":acc,"feature_importances":fi,"tree_depth":int(dt.get_depth()),
       "n_hits":int((y==1).sum()),"n_nonhits":int((y==0).sum()),"tree_text":export_text(dt,feature_names=names)}
json.dump(task2, open(os.path.join(OUT,'extracted_rules.json'),'w'), indent=2)
print(f"\n=== TASK 2 (decision tree) | n_active={task2['n_hits']} n_inactive={task2['n_nonhits']} | fidelity={acc:.4f} ===")
for k,v in sorted(fi.items(), key=lambda x:-x[1]):
    if v>0: print(f"  {k}: {v*100:.1f}%")

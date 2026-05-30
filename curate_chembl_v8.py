"""
Item 2: curate additional ChEMBL targets via the REST API (fast, large pages).
Pipeline matches the paper: IC50/Ki/EC50/Kd in nM with '=' or '~' relation,
converted to pIC50 = 9 - log10(nM), clipped to [-2,14], canonicalized and
deduplicated by canonical SMILES (median across replicates).
Writes data/<name>_chembl.csv with columns SMILES,pIC50.

Expanded panel (16 targets total with the original 4) across families:
  lipid enzymes:    faah   (CHEMBL2243)            [+ scd1, fads]
  other enzymes:    ptgs2  (CHEMBL230) COX-2; ache (CHEMBL220)
  aminergic GPCRs:  drd3   (CHEMBL234); htr2a (CHEMBL224)   [+ drd2]
  opioid GPCRs:     oprm1  (CHEMBL233); oprk1 (CHEMBL237)
  cannabinoid GPCR: cnr2   (CHEMBL253) CB2
  adenosine GPCR:   adora2a(CHEMBL251)
  kinases:          egfr   (CHEMBL203); kdr   (CHEMBL279) VEGFR2
  ion channel:      herg   (CHEMBL240)
  [peptide GPCR nk1r is already curated]
"""
import sys, csv, time, json, urllib.request
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

TARGETS = {
    'drd3': 'CHEMBL234', 'htr2a': 'CHEMBL224', 'oprm1': 'CHEMBL233',
    'oprk1': 'CHEMBL237', 'cnr2': 'CHEMBL253', 'adora2a': 'CHEMBL251',
    'faah': 'CHEMBL2243', 'ptgs2': 'CHEMBL230', 'ache': 'CHEMBL220',
    'egfr': 'CHEMBL203', 'kdr': 'CHEMBL279', 'herg': 'CHEMBL240',
}
BASE_URL = ('https://www.ebi.ac.uk/chembl/api/data/activity.json'
            '?target_chembl_id={cid}&standard_type__in=IC50,Ki,EC50,Kd'
            '&standard_units=nM&limit=1000&offset={off}')
CAP = 16000  # max records scanned per target (bounds runtime)


def fetch(cid, cap=CAP):
    off = 0; acts = []
    while off < cap:
        url = BASE_URL.format(cid=cid, off=off)
        for attempt in range(4):
            try:
                r = json.load(urllib.request.urlopen(url, timeout=90)); break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3)
        page = r['activities']
        acts.extend(page)
        total = r['page_meta']['total_count']
        off += 1000
        if off >= total or not page:
            break
    return acts, total


def curate(name, cid, outdir='data'):
    t0 = time.time()
    acts, total = fetch(cid)
    recs = {}; kept = 0
    for a in acts:
        v = a.get('standard_value'); smi = a.get('canonical_smiles')
        rel = a.get('standard_relation')
        if v is None or smi is None or rel not in (None, '=', '~'):
            continue
        try: v = float(v)
        except (TypeError, ValueError): continue
        if v <= 0: continue
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        p = 9.0 - np.log10(v)
        if p < -2 or p > 14: continue
        recs.setdefault(Chem.MolToSmiles(m), []).append(p)
        kept += 1
    rows = [(c, float(np.median(ps))) for c, ps in recs.items()]
    path = f'{outdir}/{name}_chembl.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['SMILES', 'pIC50'])
        for c, p in rows:
            w.writerow([c, f'{p:.4f}'])
    print(f"  {name} ({cid}): total={total} scanned={len(acts)} kept={kept} "
          f"-> {len(rows)} unique [{time.time()-t0:.0f}s]", flush=True)
    return len(rows)


if __name__ == '__main__':
    sel = sys.argv[1:] if len(sys.argv) > 1 else list(TARGETS)
    for nm in sel:
        try:
            curate(nm, TARGETS[nm])
        except Exception as e:
            print(f"  FAILED {nm}: {type(e).__name__}: {str(e)[:140]}", flush=True)
    print("CURATION DONE", flush=True)

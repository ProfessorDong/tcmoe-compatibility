# Target-Conditioned Multi-View MoE and Source–Target Compatibility in Few-Shot Bioactivity Prediction

Code and data accompanying the IEEE J-BHI submission *"Target-Conditioned Multi-View Mixture-of-Experts and Source–Target Compatibility in Few-Shot Bioactivity Prediction"*. Author: **Liang Dong** (`liangdng@gmail.com`, `Liang_Dong@baylor.edu`).

## Two contributions

1. **Target-conditioned multi-view Mixture-of-Experts predictor** with sparse top-K routing over four molecular views (Morgan fingerprints, a dual GAT/BiGRU cross-attention encoder embedding, MACCS keys, RDKit descriptors), trained jointly across targets with a learnable target-ID embedding.

2. **A quantitative empirical law** for when few-shot bioactivity transfer helps, fit over 240 ordered source→target pairs across 16 ChEMBL targets in nine pharmacology sub-families:

   ```
   gain(A→B) ≈ 0.36·C_nn(A→B) − 0.14·log10(n_B) + 0.48
   ```

   Both coefficients are highly significant (parametric `p = 2×10⁻⁴` and `p = 5×10⁻⁸`), robust under two-way cluster-robust standard errors, a 5,000-sample permutation test, and a support-only recomputation of `C_nn`. On small-data held-out targets (`n < 2000`), nearest-neighbor structural compatibility `C_nn` alone explains roughly half the variance in transfer gain (Pearson `r = 0.72`, `p = 7×10⁻⁶`). Because `C_nn` is computed from SMILES alone, the metric is an **operational, prospective pre-screen** for selecting source datasets for any new low-data target.

## Repository layout

```
tcmoe-compatibility/
├── README.md                          # this file
├── LICENSE                            # MIT
├── requirements.txt
├── verify_tables.py                   # programmatic regression test (93 assertions)
├── run_moe_predictor.py               # MoE training/eval entry
├── run_compat_v8.py                   # 240-pair pairwise compatibility experiment
├── run_compat_dual_spotcheck.py       # dual-encoder probe-architecture spot-check
├── run_fusion_ablation_v7.py          # encoder fusion ablation
├── run_3view_v7.py                    # 3-view MoE (descriptor expert dropped)
├── curate_chembl_v8.py                # ChEMBL data curation (12 additional targets)
├── analyze_compat_v8.py               # main compat analysis + scatter figure
├── analyze_compat_cv.py               # cross-family LOO + permutation test
├── analyze_compat_rigor.py            # support-only C_nn + two-way cluster-robust SEs
├── models/
│   ├── dual_encoder.py                # DualEncoder (pre-pooling cross-attention + gated fusion)
│   ├── moe_predictor.py               # TargetConditionedRouter + MoEPredictor
│   ├── property_predictor.py
│   ├── graph_encoder.py               # GAT
│   └── smiles_encoder.py              # BiGRU
├── data/                              # 16 curated target CSVs (SMILES, pIC50)
│   ├── scd1_binding.csv               # primary target 1 (n=762)
│   ├── fatty_acid_desaturase_bioactivity.csv  # FADS (n=1187)
│   ├── nk1r_combined.csv              # NK1R (n=3056)
│   ├── drd2_bioactivity.csv           # DRD2 (n=9966)
│   ├── faah_chembl.csv  ...           # 12 additional ChEMBL targets
│   └── dataset.py, featurizer_views.py, ...   # data loaders
└── outputs/                           # frozen JSON results reproducing every cited number
    ├── moe_v6/                        # within-target ablations, selectivity, top-K, factorial
    ├── moe_v7/                        # few-shot LOTO (per-target AGG over 5×5 runs)
    ├── fusion_ablation_v7/            # encoder fusion ablation
    ├── compat_v8_16tgt.json           # 240-pair compatibility raw results
    ├── compat_v8_16tgt_supC.json      # support-only C_nn sensitivity
    ├── compat_dual_spotcheck.json     # probe-architecture spot-check
    └── moe_3view_results.json         # 3-view ablation
```

## Reproducing every cited number

```bash
pip install -r requirements.txt
python verify_tables.py        # 93 programmatic assertions; non-zero exit on mismatch
```

`verify_tables.py` is the canonical regression test — it asserts every numeric claim in the paper against the frozen JSONs under `outputs/` and returns non-zero if any cell drifts.

### Re-running individual experiments

| Output | Producer | Approximate compute |
|---|---|---|
| Few-shot LOTO (per target) | `python run_moe_predictor.py --experiment fewshot --held_out <scd1\|fads\|nk1r\|drd2>` | a few hours per target on a single GPU (5 draws × 5 resamplings × 4 k × 8 methods) |
| Within-target + selectivity + top-K | `python run_moe_predictor.py --experiment all` | a couple of hours on a single GPU |
| Encoder fusion ablation | `python run_fusion_ablation_v7.py` | ~30 min |
| 240-pair compatibility study | `python run_compat_v8.py --targets scd1 fads nk1r drd2 drd3 htr2a oprm1 oprk1 cnr2 adora2a faah ptgs2 ache egfr kdr herg --out outputs/compat_v8_16tgt.json` | ~1.5 h |
| Cross-family LOO + permutation | `python analyze_compat_cv.py outputs/compat_v8_16tgt.json` | <1 min |
| Support-only C_nn + cluster-robust SEs | `python analyze_compat_rigor.py` | ~5 min |
| Dual-encoder probe spot-check | `python run_compat_dual_spotcheck.py` | ~30 min |
| Compatibility-gain scatter | `python analyze_compat_v8.py outputs/compat_v8_16tgt.json compat_gain.eps` | <10 s |
| 12 new ChEMBL target CSVs | `python curate_chembl_v8.py` | ~5 min, requires internet |

GPU is an NVIDIA RTX 4060 (8 GB); the code falls back to CPU when CUDA is unavailable.

## Datasets

The primary four targets (SCD-1, FADS, NK1R, DRD2) and twelve additional ChEMBL targets (DRD3, 5HT2A, OPRM1, OPRK1, CNR2, ADORA2A, FAAH, PTGS2, AChE, EGFR, KDR, hERG) are pooled IC₅₀/Ki/EC₅₀/Kd records from ChEMBL release 33, converted to `pIC50 = 9 − log10(nM)`, clipped to `[−2, 14]`, canonicalized with RDKit, and median-deduplicated by canonical SMILES (with the exception of the SCD-1 literature-curated portion, which retains replicate measurements as separate rows). Compound counts span 762 (SCD-1) to 9,966 (DRD2), median 6,504. All ChEMBL IDs and the curation pipeline are in `curate_chembl_v8.py`.

The data are released under the same terms as ChEMBL (CC BY-SA 3.0 for the ChEMBL-derived portion).

## License

Code: MIT (see `LICENSE`).
Data: derived from ChEMBL release 33, distributed under CC BY-SA 3.0.

## Contact

Liang Dong &lt;liangdng@gmail.com&gt;, &lt;Liang_Dong@baylor.edu&gt; — Department of Electrical and Computer Engineering, Baylor University, and Department of Radiology, UT Southwestern Medical Center.

This work was supported by NCI/NIH award R01CA309499.

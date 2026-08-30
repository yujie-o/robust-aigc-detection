# P2: Generalisation

Tests UnivFD's Nearest Neighbor classifier against the baseline (UnivFD Linear Classification) on frozen SigLIP2 features. See `results.md` for full results and analysis.

## File structure
```
experiments/generalisation/
│
├── README.md                         
├── results.md                        
├── __init__.py
├── nn_classifier.py                  
├── nn_classifier_model.py             
├── save_nn_checkpoint.py              
├── organiser_eval_loader.py           
├── run_p2_evaluation.py               
│
├── data/
│   ├── dalle3.csv                     
│   └── real_coco.csv                  
│
└── results/
    ├── feature_bank.pt                
    ├── val_features.pt                           
    ├── Baseline_conditions.json       
    └── P2a-NN_conditions.json         
```

## What each file does

| File | Purpose |
|---|---|
| `nn_classifier.py` | Extracts SigLIP2 features from SID training + val, saves feature bank, runs NN classification. |
| `nn_classifier_model.py` | Wraps NN classifier as `nn.Module` so it works with the shared evaluator. Feature bank stored as buffers. |
| `save_nn_checkpoint.py` | Loads saved feature bank, packages into `checkpoints/p2a_nn_best.pt`. |
| `organiser_eval_loader.py` | Reads CSVs in `data/`, filters organiser eval subset, downloads images from ModelScope, caches locally. |
| `run_p2_evaluation.py` | Runs P2-a checkpoint through shared evaluator (internal + external + all transformations). |

## Reproducing

From repo root:

```bash
python experiments/generalisation/nn_classifier.py           
python experiments/generalisation/save_nn_checkpoint.py      
python experiments/generalisation/run_p2_evaluation.py \
    --checkpoint checkpoints/p2a_nn_best.pt --model-name P2a-NN   
```

Requires `checkpoints/baseline_best.pt` from P1's `src/train.py`.

## Dependencies

- `src/data/sid_dataset.py`, `src/models/baseline.py` (P1)
- `src/evaluation/evaluator.py`, `src/evaluation/transforms.py` (P5)
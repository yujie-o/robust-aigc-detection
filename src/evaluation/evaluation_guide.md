# Common Evaluator — Guide

How to plug your checkpoint into the shared evaluator.
You shouldn't need to edit `evaluator.py` or `transforms.py`.


## 1. Local Wildfake Dataset Setup (Mounting to `raw_data/`)

To avoid network timeouts and streaming latency during evaluation, the external benchmark
reads directly from a local `raw_data/` folder at your project root.

Create the folder and place your images in this structure:

```
robust-aigc-detection/
raw_data/
└── wildfake/
    ├── val2017/
    └── dalle/ 
```

## 2. Your model

The evaluator calls your model like this:

```python
inputs = processor(images=batch_of_PIL_images, return_tensors="pt")
logits = model(inputs["pixel_values"].to(device))  
# single tensor in, logits out
probs = torch.sigmoid(logits)
```

So your model needs to:

1. Take a **single** `pixel_values` tensor in `forward()` and return **logits** (pre-sigmoid).
   See `models/baseline.py` for the reference shape.
2. Save checkpoints as `{"model_state_dict": ...}` (same as `train.py` does) so
   `load_model_for_eval` can load them.
3. Use the default `AutoProcessor` for `MODEL_NAME` (SigLIP2).


## 3. Run it

example:
```bash
PYTHONPATH=src python src/evaluation/run_evaluation.py \
  --checkpoint checkpoints/baseline_best.pt \
  --model-name Baseline \
  --samples-per-class 100 \
  --batch-size 32
```

### CLI Options

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *(Required)* | Filepath to your `.pt` model checkpoint. |
| `--model-name` | *(Required)* | Label for the run (`Baseline`, `P2`, `P3`, `P4`). |
| `--samples-per-class` | `None` | Number of balanced samples per class to evaluate. (use none so that all external datasets are used) |
| `--batch-size` | `32` | Inference batch size (increase to 32/64 to speed up evaluation). |

This loads your checkpoint, evaluates it on the shared internal SID_Set validation split across all transform conditions, then runs a clean-condition pass on the organiser's WildFake external set (NEVER train on it without removing COCO val2017 and DALL·E Advanced).

**If your model class isn't `AIGCDetector`:** copy `run_evaluation.py` and pass your class into
`load_model_for_eval(checkpoint, device, model_class=YourModel)`. Everything else stays the same.

### Evaluation Modes

`main()` runs one of two evaluation functions. Pick the mode by commenting/uncommenting
the corresponding line at the bottom of `run_evaluation.py`:

```python
run_full_combined_evaluation(model, device, args)
#run_external_only_evaluation(model, device, args)
```

| Mode | Function | What it does | Output file |
|---|---|---|---|
| **Combined** (default) | `run_full_combined_evaluation` | Runs all transform conditions on the **internal SID_Set held-out split**, then a single clean-condition pass on **external WildFake**. Reports the internal/external AUC gap (`internal_external_gap`). | `results/<model-name>_conditions.json` |
| **External-only** | `run_external_only_evaluation` | Runs all transform conditions on **WildFake only** — no internal set, no gap metric. Useful for a quicker external-generalisation check. | `results/<model-name>_external_conditions.json` |

Both modes append the same row shape to `results/summary_table.md`, so results from either
mode land in the same shared table — just switch which function is active before running.


## 4. What you get

- `results/<model-name>_conditions.json`: every prediction, for error analysis
- `results/summary_table.md`: one shared row per model:

| Model | Clean | JPEG | Blur | Resize | Noise | Color | Crop | Robust Mean | External |
|---|---|---|---|---|---|---|---|---|---|


## 5. Error analysis 

- **False positives**: call `get_false_positives_negatives(predictions, threshold=0.5)` (not yet called in `run_evaluation.py`)
- **False negatives**: same function as above (`get_false_positives_negatives`), also unused.
- **Hardest transformation**: `find_hardest_condition(condition_results)`, also unused.
- **Biggest performance degradation**: `find_biggest_degradation(summary)` / `compute_degradation(summary)`,
  saved under `error_analysis.biggest_degradation_group` / `error_analysis.degradation_by_group` in the JSON.
- **JPEG/compression bias**: `jpeg_trend(condition_results)`, saved under `error_analysis.jpeg_trend`.
- **Dataset-specific signals vs. real AIGC cues**:  `internal_external_gap(summary, external_auc)`, saved under `error_analysis.internal_external_gap` (internal clean AUC minus external WildFake AUC).

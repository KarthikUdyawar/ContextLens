# Models — v1.0.0

## Architecture

`CustomBERTClassifier` (`src/utils/custom_BERT_classifier.py`):

```
bert-base-uncased (pretrained, fine-tuned)
  → CLS token embedding (768-d)
  → Linear(768, 128) → ReLU → Dropout(0.2)
  → Linear(128, 64)  → ReLU → Dropout(0.2)
  → Linear(64, 3)
  → Softmax(dim=1)
```

3 output classes: negative / neutral / positive.

## Checkpoints in repo

| Version | Location                                       | Notes                                                                                                                       |
| ------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 0.1v    | `src/model/0.1v/checkpoint_0` … `checkpoint_6` | 7 checkpoints, one per training run/epoch batch. Reports + confusion matrices present, no `model.pth` (weights gitignored). |
| 0.2v    | `src/model/0.2v/`                              | Latest, single checkpoint. Reports present, weights gitignored.                                                             |

No committed weights means **no model ships with the repo** — this is the most severe gap for anyone trying to run the API out of the box (see DECISIONS.md #7).

## Known correctness issues

1. **Double softmax.** The model's `forward()` already applies `Softmax`. `predict.py::classify_sentiment` calls `torch.softmax()` again on the output. Argmax label is unaffected (softmax is monotonic), but reported probabilities in `/predict-prob` are miscalibrated.
2. **Loss/activation mismatch.** Training uses `BCEWithLogitsLoss` (expects raw logits, treats classes independently) against a model whose output is already softmax-normalized (mutually-exclusive, bounded [0,1], sums to 1). Standard practice for single-label 3-class classification is `CrossEntropyLoss` on raw logits with no softmax in the model. Current setup can still learn but the loss landscape doesn't match the stated single-label problem.
3. **Labels are TextBlob polarity heuristics**, not human-annotated ground truth (see PIPELINE.md). Reported accuracy (gate: `>0.85`) is measured against this same heuristic, not an independent test set.

## Acceptance gate

`build_model.py` only saves a checkpoint if `test_accuracy > 0.850` against the TextBlob-labeled test split. No held-out human-labeled benchmark exists in the repo.

# Lab 1 — Neural Network Fundamentals with TensorFlow & Keras

Completed notebooks for Lab 1. Work through them in order; each builds on the last.

| Notebook | Covers |
|---|---|
| `part1_layers_and_sequential_COMPLETED.ipynb` | Tensors, layer types, the Sequential API, Dropout + EarlyStopping, first MNIST classifier |
| `part2_functional_api_COMPLETED.ipynb` | Where Sequential runs out, the Functional API, a multi-input (image + noisy OCR text) model, branching topologies |
| `part3_full_pipeline_COMPLETED.ipynb` | Full pipeline with Weights & Biases experiment tracking, three tracked runs, results table, final test-set check |

## Running them

Built for Google Colab — open each notebook there and run top to bottom.
Dependencies (TensorFlow, pandas, matplotlib, wandb) are either preinstalled
or installed by the notebook.

### Weights & Biases (Part 3 only)

Part 3 logs every run to W&B. Provide your API key **without hardcoding it**:

- **In Colab:** left sidebar → key icon → add a secret named `WANDB_API_KEY`.
- **Locally:** put `WANDB_API_KEY=...` in a `.env` file (gitignored), or run
  `wandb login` once.

If no key is found, the notebooks still run end to end — nothing is tracked.

# Multimodal Authentication + Product Recommendation

ALU Formative 2: Data Preprocessing.

A user is identified by **face**, a **product model** predicts which product category they would buy, and the prediction is only shown after **voice** verification confirms it is the same person.

```
face image -> face model -> known member? --no--> ACCESS DENIED
                               | yes
                               v
                    product model (result held back)
                               |
voice clip -> voice model -> same member? --no--> ACCESS DENIED
                               | yes
                               v
                   show product recommendation
```

## Repo layout

```
data/
  raw/                 original datasets (never edited)
  processed/           merged + feature-engineered customer data
  images/<member>/     face photos, 3+ per member (see data/NAMING.md)
  audio/<member>/      voice clips, 2+ per member
  unauthorized/        non-member face + voice, for the access-denied demo
features/              image_features.csv, audio_features.csv
models/                trained models
notebooks/             main walkthrough notebook
src/                   pipeline scripts + command-line app
reports/figures/       saved plots for the report
```

## How to run (from the repo root)

```bash
pip install -r requirements.txt
python src/product_model.py          # merge data, train + evaluate the product model
python src/image_features.py         # image_features.csv, train + evaluate the face model
python -m src.audio_pipeline         # audio_features.csv, train + evaluate the voice model
python src/demo.py --member towi --other kethia --outsider towi   # all 4 demo scenarios
python src/app.py --face data/images/towi/towi_neutral.jpg --voice data/audio/towi/towi_yes_approve.wav
```

`demo.py` runs: (1) approved transaction, (2) outsider face denied, (3) outsider voice denied,
(4) right face with another member's voice denied. `app.py` exits with code 0 when access is granted, 1 when denied.

Every model is evaluated with accuracy, macro F1 and log loss, holding out whole originals
(a customer, a photo or a recording) so augmented copies never leak into the test fold.
Metrics are saved in `models/*_metrics.json` and shown in the notebooks.

## Pipeline

| Step | Output | Owner |
|---|---|---|
| Clean + merge tabular data, EDA | `data/processed/merged_customer_data.csv` | Tabular lead |
| Image augmentation + feature extraction | `features/image_features.csv` | Image lead |
| Audio augmentation + feature extraction | `features/audio_features.csv` | Audio lead |
| Train + evaluate face, voice, product models | `models/` | Each lead |
| Command-line simulation (approved + unauthorized) | `src/` app | Integration lead |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Working together

- One branch per role: `feat/tabular`, `feat/image`, `feat/audio`, `feat/integration`. Open a pull request into `main`; one teammate reviews.
- Agree on CSV column names before writing code, and don't rename them afterwards.
- In the notebook, edit only your own section and clear large outputs before pushing.

## Team contributions

| Member | Role | Main work |
|---|---|---|
| | Tabular lead | |
| | Image lead | |
| | Audio lead | |
| | Integration lead | |
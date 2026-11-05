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

## Planned pipeline

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
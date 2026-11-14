# Multimodal Authentication + Product Recommendation

ALU Formative 2: Multimodal Data Preprocessing. Group 32.

A user is identified by **face**, a **product model** predicts which product category they would buy, and the prediction is only shown after **voice** verification confirms it is the same person. Any failed check means access denied.

- **Report:** see the PDF submitted on Canvas
- **Demo video:** _(link)_
- **Contribution tracker:** [Google Sheet](https://docs.google.com/spreadsheets/d/15F1ADVbkytfpCQRSzjRjeGjX7Jv60VrhuZOJOFpeyZ4/edit?usp=sharing)

## How it works

```
face image -> face model -> member with confidence >= 0.94? --no--> ACCESS DENIED
                               | yes: this picks the member's customer profile
                               v
                    product model runs (result held back)
                               |
voice clip -> voice model -> SAME member, confidence >= 0.41? --no--> ACCESS DENIED
                               | yes
                               v
                   show product recommendation
```

- **Face first** decides who the user is. **Voice last** confirms it, right before anything is revealed.
- The voice must name the **same member** as the face, so one member's face plus another member's voice is rejected.
- Both identity models always name one of the four members, even for a stranger, so each has a **confidence threshold** tuned on outsider samples that were never used for training.
- The datasets describe customers, not us, so each member is linked to one real customer in `src/config.py` (`MEMBER_TO_CUSTOMER`: Kethia 101, Tabitha 114, Towi 152, Harmony 188).

## Results

All scores are on held-out data. Each evaluation holds out whole originals (a customer, a photo or a recording) together with all their augmented copies, so near-copies never sit in both training and testing.

| Model | Data | Algorithm | Evaluation | Accuracy | Macro F1 | Log loss |
|---|---|---|---|---|---|---|
| Face recognition | 24 photos, 96 rows | MobileNetV2 embeddings + logistic regression | Leave one photo out | 1.00 | 1.00 | 0.005 |
| Voiceprint verification | 16 clips, 96 rows | 47 audio features + random forest | Leave one recording out | 62.5% | 0.65 | 0.88 |
| Product recommendation | 117 purchases, 61 customers | Random forest | 5-fold CV by customer, 3 seeds | 26.5% | 0.21 | 1.63 |

For reference: guessing among 4 members scores 25% and log loss 1.39; the product baseline (most common category) scores 21.9%. A permutation test on the product model gives p = 0.84, so the tabular data carries almost no signal; see the team notebook, section 2.5.

## Quick start

Tested on Windows with Python 3.13 (Anaconda). Run every command from the repo root.

```bash
pip install -r requirements.txt
```

On Windows, if `python` opens the Microsoft Store version, point it at the Python you installed the packages into, for example in PowerShell:

```powershell
Set-Alias python "$env:USERPROFILE\anaconda3\python.exe"
```

### Run the system

```bash
# all four demo scenarios, with an OK / !! summary
python src/demo.py --member towi --other kethia --outsider towi

# one transaction with your own files
python src/app.py --face data/images/towi/towi_neutral.jpg --voice data/audio/towi/towi_yes_approve.wav

# interactive: asks for the two file paths, then shows what each model sees and every probability
python src/app.py --explain
```

`demo.py` runs: (1) approved transaction, (2) outsider face, denied at step 1, (3) outsider voice, denied at step 3, (4) right face with another member's voice, denied at step 3. `app.py` exits with code 0 when access is granted and 1 when denied. The first run takes 10 to 20 seconds while TensorFlow loads.

### Rebuild features and retrain the models

Each command rebuilds its outputs from the raw files, evaluates, refits the final model on all the data and saves it to `models/`. The app loads the new files on its next run.

```bash
python src/data_merge.py          # clean + merge -> data/processed/merged_customer_data.csv
python src/product_model.py       # compare models, evaluate, save product model + metrics
python src/image_features.py      # crop, augment, embed -> features/image_features.csv, train face model
python -m src.audio_pipeline      # preprocess, augment, extract -> features/audio_features.csv, train voice model
python src/product_model.py --customer 152    # one prediction from the saved product model
```

The face and voice runs reproduce the saved files exactly (fixed random seeds). The saved product model was trained with scikit-learn 1.9.0; other scikit-learn versions split the cross-validation folds differently and give slightly different numbers, so retrain it with 1.9.0 to match the report.

## Notebooks

| Notebook | What it shows |
|---|---|
| `notebooks/multimodal_auth_system.ipynb` | The whole project: tabular EDA, cleaning, merge validation and the product model in full (sections 1 and 2), summaries of the face, voice and multimodal parts (sections 3 to 5) |
| `notebooks/image_processing.ipynb` | Face steps: photo inventory, crop check, augmentation, embeddings, model evaluation, threshold |
| `notebooks/audio_processing.ipynb` | Voice steps: inventory, preprocessing, waveforms and spectrograms, augmentation, features, model evaluation, threshold |

Outputs are saved, so the notebooks can be read without running them. Figures are also saved in `reports/figures/`.

## Data

| Data | What | Where |
|---|---|---|
| Customer tables | `customer_social_profiles.xlsx` (155 x 5), `customer_transactions.xlsx` (150 x 6) | `data/raw/` |
| Merged dataset | 117 rows x 29 columns, one row per purchase | `data/processed/merged_customer_data.csv` |
| Face photos | 4 members x 3 expressions (neutral, smiling, surprised) x 2 takes = 24 JPGs | `data/images/<member>/<member>_<expression>[_2].jpg` |
| Voice clips | 4 members x 2 phrases ("Yes, approve", "Confirm transaction") x 2 takes = 16 WAVs | `data/audio/<member>/<member>_<phrase>[_2].wav` |
| Outsider samples | 2 photos and 2 clips from one person outside the group, never used for training | `data/unauthorized/images/`, `data/unauthorized/audio/` |
| Image features | 96 rows: member, expression, augmentation, source_image, 1,280 embedding values | `features/image_features.csv` |
| Audio features | 96 rows: member, phrase, augmentation, source_audio, 47 features (MFCCs, roll-off, energy, centroid, zero-crossing rate) | `features/audio_features.csv` |

Naming rules are in `data/NAMING.md`.

## Pipeline and key decisions

**Tabular** (`src/data_merge.py`, `src/product_model.py`)
- Dropped 5 duplicate social rows; matched IDs stored as `A178` and `178`; filled 10 missing ratings with the overall median plus a `rating_missing` flag (not the per-category median, which would leak the target).
- Both tables have several rows per customer, so a direct join turns 150 purchases into 213 rows. Social data is aggregated to one row per customer first, then **inner joined** so every row has real social features. 8 post-merge checks pass.
- Logistic regression, random forest and gradient boosting compared against two baselines, split by customer so the model can't memorise people.

**Face** (`src/image_features.py`)
- OpenCV Haar cascade crops the face (so the model learns faces, not backgrounds), resized to 224 x 224.
- 3 augmentations per photo: flip, 15 degree rotation, grayscale.
- MobileNetV2 pre-trained on ImageNet gives 1,280 numbers per face (24 photos are too few to train a CNN), then scaling + logistic regression.
- Threshold 0.94: held-out member photos score 0.964 or more, outsider photos 0.69 and 0.86.

**Voice** (`src/audio_pipeline.py`)
- Every clip resampled to 16 kHz mono, silence trimmed at 20 dB, peak-normalised, so the model learns voices rather than phones or distance.
- 5 augmentations per clip: pitch +/-1 semitone, speed x1.1 and x0.9, background noise at 20 dB SNR.
- 47 features summarised by mean and std over 32 ms frames; random forest beat logistic regression (56.3%, F1 0.56).
- Threshold 0.41, chosen by balanced accuracy against outsider clips.

**App** (`src/app.py`, `src/demo.py`): chains `recognise_face`, `predict_for_customer` and `verify_speaker`. All paths and thresholds come from `src/config.py`.

## Repo layout

```
data/
  raw/                 original datasets (never edited)
  processed/           merged + feature-engineered customer data
  images/<member>/     face photos (see data/NAMING.md)
  audio/<member>/      voice clips
  unauthorized/        outsider photos and clips, for the access-denied demo only
features/              image_features.csv, audio_features.csv
models/                face_model.joblib, voice_model.joblib, product_model.joblib + metrics JSON files
notebooks/             team notebook, image notebook, audio notebook
src/
  config.py            shared paths, thresholds, member-to-customer mapping
  data_merge.py        tabular cleaning and merge
  product_model.py     product model training and predict_for_customer()
  image_features.py    face pipeline, training and recognise_face()
  audio_pipeline.py    voice pipeline, training and verify_speaker()
  app.py               command-line app (face -> product -> voice)
  demo.py              runs the four demo scenarios
reports/figures/       saved plots used in the report
```

## Limitations

- Few samples per person (6 photos, 4 clips), so every score is a rough estimate; one voice mistake moves accuracy by about 6 points.
- Each person's photos come from one or two sessions, so the perfect face score will drop in new settings.
- One outsider only, used to set the thresholds, with nothing left for a hidden final test.
- The product data has almost no signal (p = 0.84), so recommendations are best guesses; the app shows every probability.

## Team contributions

| Member | Role | Main work |
|---|---|---|
| Oluwaloseyitowi Iji | Audio lead | Repository structure; audio pipeline (`src/audio_pipeline.py`, `notebooks/audio_processing.ipynb`): preprocessing, augmentation, `audio_features.csv`, voice model, threshold and `verify_speaker`. Final integration and testing of the full pipeline. |
| Harmony Naomi Niyongabo | Integration lead | Command-line app (`src/app.py`) with the face, product, voice decision chain and clear "access denied" messages; `src/demo.py` with the approved and unauthorized scenarios. |
| Kethia Ngabire Kayigire | Tabular lead | Tabular pipeline (`src/data_merge.py`, `src/product_model.py`): cleaning, aggregation, inner join, `merged_customer_data.csv`, model comparison, customer-grouped evaluation, permutation test and `predict_for_customer`. Shared `src/config.py`; notebook sections 1 and 2. |
| Tabitha Dorcas Akimana | Image lead | Image pipeline (`src/image_features.py`): face crop, augmentation, MobileNetV2 features, `image_features.csv`, face model and `recognise_face`; the image notebook steps. Found the OpenCV 5 problem and its fix. |

Everyone contributed face photos and voice clips.

# Data naming rules

Use your lowercase first name in every file name.

## Images -> `data/images/<name>/`

| File | What |
|---|---|
| `<name>_neutral.jpg` | relaxed face, mouth closed |
| `<name>_smiling.jpg` | clear smile |
| `<name>_surprised.jpg` | eyebrows up, mouth open |

- JPG, front-facing, plain background, good light, no filters, same phone for all three.
- Extra takes help the face model: `<name>_neutral_2.jpg`, etc.

## Audio -> `data/audio/<name>/`

| File | Say |
|---|---|
| `<name>_yes_approve.wav` | "Yes, approve" |
| `<name>_confirm_transaction.wav` | "Confirm transaction" |

- WAV if possible (m4a is fine, we convert). Quiet room, phone about 20 cm away, 2 to 3 seconds.
- Extra takes help the voice model: `<name>_yes_approve_2.wav`.

## Unauthorized samples -> `data/unauthorized/`

One face and one voice clip from someone outside the group (with their permission), e.g.
`unauthorized/images/stranger1_neutral.jpg`, `unauthorized/audio/stranger1_yes_approve.wav`.
Never used for training, only to test the access-denied path.
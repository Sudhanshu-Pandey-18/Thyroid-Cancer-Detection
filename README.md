```markdown
# Thyroid Recurrence Predictor

A Streamlit app that encodes clinical inputs with saved label encoders and predicts thyroid cancer recurrence with a probability score.

## Features
- Interactive form for key fields (Age, Gender, Pathology, T/N/M, Stage, Response, etc.).
- Uses pre-fitted label encoders to match the training schema.
- Displays predicted class and probability of recurrence.

## Quickstart
- Python 3.x
- Install:
  ```
  pip install streamlit pandas joblib
  ```
- Configure model paths in `app.py`:
  ```
  model = joblib.load("model/thyroid_recurrence_model.pkl")
  encoders = joblib.load("model/label_encoders.pkl")
  target_encoder = encoders["Recurred"]
  ```
  Place both files in `model/`.

- Run:
  ```
  streamlit run app.py
  ```

## How it works
- Builds a single-row DataFrame from inputs with column names matching training time.
- Applies the corresponding label encoder for each encoded column before prediction.
- Predicts class and probability; inverts target class with the encoder key `Recurred`.

## Field name notes
- Column and encoder keys must match exactly, including spaces (e.g., `Hx Radiothreapy`, `Physical Examination`).
- `label_encoders.pkl` should include one encoder per categorical feature plus `Recurred` for inverse-transform.

## Suggested structure
```
.
├─ app.py
├─ model/
│  ├─ thyroid_recurrence_model.pkl
│  └─ label_encoders.pkl
├─ requirements.txt
└─ README.md
```
```

# End-to-End ML Project: Student Performance Prediction

Predicts a student's **math score** from demographic details and their reading and writing scores, using a modular training pipeline and a Flask web app, with configuration for AWS Elastic Beanstalk deployment.

> **Credit:** This project was built by following Krish Naik's end-to-end ML project tutorial ([krishnaik06/mlproject](https://github.com/krishnaik06/mlproject)). This version includes the bug fixes listed below.

## Results

| Item | Value |
| --- | --- |
| Dataset | Students Performance, 1,000 rows (`notebook/data/stud.csv`) |
| Task | Regression (target: `math_score`) |
| Models compared | Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, AdaBoost (GridSearchCV tuning) |
| Best model | Linear Regression |
| Test R² | 0.88 |

## Project Structure

```
├── app.py                         # Flask web app (form → prediction)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # reads data, train/test split (80/20)
│   │   ├── data_transformation.py # imputation, scaling, one-hot encoding
│   │   └── model_trainer.py       # trains + tunes models, picks best by R²
│   ├── pipeline/
│   │   └── predict_pipeline.py    # loads saved preprocessor + model, predicts
│   ├── exception.py               # custom exception with file/line info
│   ├── logger.py                  # timestamped logging
│   └── utils.py                   # save/load objects, model evaluation
├── notebook/                      # EDA and model training notebooks + data
├── artifacts/                     # saved model.pkl, preprocessor.pkl, splits
├── templates/                     # HTML pages for the web app
└── .ebextensions/                 # AWS Elastic Beanstalk config
```

## How to Run

```bash
pip install -r requirements.txt

# Train: runs ingestion → transformation → model training
python -m src.components.data_ingestion

# Start the web app, then open http://localhost:5000/predictdata
python app.py
```

## Fixes in This Version

- **Swapped inputs:** the web form passed the writing score as the reading score and vice versa; corrected the mapping in `app.py`.
- **Preprocessor file mismatch:** training saved `proprocessor.pkl` (typo) while prediction loaded `preprocessor.pkl`, so the app used a stale preprocessor. Unified the file name.
- **Missing import:** added `import os` in `predict_pipeline.py`, which crashed every prediction.
- **Windows-only path:** replaced `'notebook\data\stud.csv'` with `os.path.join(...)` so training runs on Linux, Docker and AWS.
- **pandas compatibility:** removed the conflicting `axis=1` from `drop(columns=...)`, which errors on recent pandas.
- **Missing dependency:** added `dill` to `requirements.txt`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, Flask, AWS Elastic Beanstalk

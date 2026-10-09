# Titanic MLOps Pipeline

An end-to-end MLOps project: data preprocessing, model training and comparison, experiment tracking with MLflow, data versioning with DVC, CI/CD with GitHub Actions, a secured FastAPI prediction service in Docker, and monitoring with Evidently AI.

## About this project

- This is a course project for Unit 5 (MLOps) of an AI/ML program. The code carries line-by-line comments written as learning notes, by design.
- **Scenario note:** the assignment describes predicting 30-day hospital readmission, but the dataset provided is the Titanic passenger dataset. The model predicts `Survived`. No healthcare data was created or simulated. Where the readmission framing guided a decision (for example, prioritizing recall), the documentation says so.

## Project structure

| File or folder | Purpose |
|---|---|
| `exploration.ipynb` | Data exploration, cleaning, feature engineering, and comparison of three models |
| `preprocess.py` | Cleans the raw data and adds features; writes `data/processed_titanic.csv` |
| `train.py` | Trains the selected model, logs it to MLflow, saves `model.pkl` and monitoring data |
| `test_model.py` | Six checks on the processed data and the trained model |
| `app.py` | FastAPI prediction service with an API key check |
| `test_api.py` | Four checks against the running API |
| `monitor.py` | Evidently drift and performance reports with alert thresholds |
| `Dockerfile`, `requirements-api.txt` | Container image for the API |
| `.github/workflows/pipeline.yml` | GitHub Actions CI/CD pipeline |
| `data/titanic.csv` | Raw dataset (891 rows) |
| `data/processed_titanic.csv.dvc` | DVC pointer to the versioned processed data |
| `reports/` | Evidently HTML reports |
| `charts/`, `screenshots/` | Exploration charts and evidence screenshots |
| `project_report.md` | Project report |

## Setup

Requires Python 3.12, Git, and Docker Desktop.

```
git clone https://github.com/wyatt-hockmeyer/titanic-mlops-pipeline.git
cd titanic-mlops-pipeline
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```
python preprocess.py
python train.py
python test_model.py
```

`train.py` trains a Logistic Regression model (StandardScaler plus LogisticRegression in a Pipeline), logs parameters, five metrics, and the model to MLflow, and registers each run as a new version of `TitanicSurvivalModel`. To view runs:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then open http://127.0.0.1:5001. Port 5001 avoids a conflict with macOS AirPlay on port 5000.

## Data versioning

DVC tracks `data/processed_titanic.csv`. Git stores the small pointer file `data/processed_titanic.csv.dvc`, which holds the data's fingerprint, so every data version is tied to the code that produced it. After changing the processed data:

```
dvc add data/processed_titanic.csv
git add data/processed_titanic.csv.dvc
git commit -m "Update processed data"
```

The raw file `data/titanic.csv` is committed to Git so the pipeline can rebuild the processed data anywhere.

## Deployment

### Run the API locally

```
python -m uvicorn app:app --reload
```

Interactive documentation: http://127.0.0.1:8000/docs

### Run the API in Docker

```
docker build -t titanic-api:v1 .
docker run --name titanic-api -p 8000:8000 titanic-api:v1
```

### Call the API

`POST /predict/` accepts passenger fields and requires an `x-api-key` header:

```
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-dev-key" \
  -d '{"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C", "Title": "Mrs", "HasCabin": 1}'
```

Response: `{"prediction": 1, "survival_probability": 0.97}`

Run all four API checks with `python test_api.py` while the API is running.

**Security:** the API key is read from the `API_KEY` environment variable. When it is not set, the labeled default `local-dev-key` is used for local testing only. A real deployment sets its own key, for example `docker run -e API_KEY=your-key -p 8000:8000 titanic-api:v1`.

## CI/CD pipeline

`.github/workflows/pipeline.yml` runs on every push to `main` that changes data, Python code, requirements, the Dockerfile, or the workflow itself. It can also be started manually from the Actions tab. Steps:

1. Install Python 3.12 and the requirements
2. Preprocess the data
3. Train and log the model
4. Run the six model tests
5. Build the Docker image
6. Start the container and run the four API tests against it
7. Generate the Evidently monitoring reports
8. Upload `model.pkl` and the reports as downloadable artifacts

Any failed step stops the run and marks it failed.

## Monitoring

```
python monitor.py
```

Builds two Evidently reports with `DataDriftPreset` and `ClassificationPreset`:

1. `reports/monitoring_report.html`: training data compared with test data (normal conditions)
2. `reports/drift_simulation_report.html`: a simulated shift (passengers 20 years older, all third class, no cabin records) to show the alerts working

After each report, a threshold check prints an `ALERT` when accuracy falls below 0.75 or more than 20% of columns drift. Evidently's own dataset drift flag waits until 50% of columns drift, so these thresholds give earlier warning.

## Results

Logistic Regression was selected over Random Forest and Gradient Boosting. Test set (179 passengers):

| Metric | Score |
|---|---|
| Accuracy | 0.832 |
| Precision | 0.800 |
| Recall | 0.754 |
| F1 | 0.776 |
| ROC-AUC | 0.872 |

The full comparison and reasoning are in `exploration.ipynb` Section 7 and in `project_report.md`.

## Reproducibility and scalability practices

- Fixed `random_state=42` and a stratified split
- Preprocessing in one script shared by the notebook logic, training, and CI
- Pinned API package versions in `requirements-api.txt`, so the container reads the model with the same scikit-learn version that created it
- Every model version registered in MLflow; every data version recorded by DVC
- Automated tests gate every change; the container is tested in CI before the model is published
- The API runs in a container, so it can be deployed to any container platform and scaled horizontally

## Known limitations

- Fill values for `Age` and `Embarked` and the `Fare` cap are calculated on all 891 rows before the train/test split. The effect on test scores is negligible but noted.
- Tree-based model results can differ by one or two passengers between machines.
- On Apple silicon Macs, NumPy 2.0.2 prints harmless `RuntimeWarning` messages during matrix math. Results are unaffected and match Linux runs exactly.
- DVC uses its local cache only; a shared remote (such as cloud storage) would be added for team use.
- GitHub reports a Node.js 20 deprecation notice for the Actions used; the runs complete successfully on the newer runtime.
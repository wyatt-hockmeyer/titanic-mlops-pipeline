# train.py
# Trains the selected Logistic Regression model, logs it to MLflow, and saves it
# Run from the project folder with: python train.py

# Import pandas for working with tables of data
import pandas as pd

# Import joblib to save the trained model to a file (Lesson 38 Demo 2)
import joblib

# Import MLflow and its scikit-learn helper (Lesson 38 Demos 1 and 3)
import mlflow
import mlflow.sklearn

# Import the split, pipeline, scaler, and model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Import the five evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Settings used for this training run
processed_path = "data/processed_titanic.csv"
test_size = 0.2
random_state = 42
max_iter = 10000

# Track whether the processed file loaded
loaded = False

# Try to load the processed data
try:
    df = pd.read_csv(processed_path)
    loaded = True
except FileNotFoundError:
    print("Error : could not find", processed_path)
    print("Run python preprocess.py first")

# Only continue if the file loaded
if loaded:
    # Separate the features (X) from the target (y)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split the data the same way as the notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Store MLflow run records in a local SQLite database file (current MLflow default)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Group all runs for this project under one experiment name
    mlflow.set_experiment("titanic-survival")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Show the run ID so it can be found in the MLflow UI
        print("Run ID :", run.info.run_id)

        # Build the pipeline: scale the features, then fit logistic regression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("log_reg", LogisticRegression(max_iter=max_iter, random_state=random_state))
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Log the parameters for this run
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("training_rows", X_train.shape[0])

        # Predict classes and probabilities for the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate the five metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Log the five metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the model and register it in the MLflow model registry
        mlflow.sklearn.log_model(
            sk_model=model,
            name="titanic_model",
            registered_model_name="TitanicSurvivalModel"
        )

        # Show the metrics
        print("Accuracy :", round(accuracy, 3))
        print("Precision :", round(precision, 3))
        print("Recall :", round(recall, 3))
        print("F1 :", round(f1, 3))
        print("ROC-AUC :", round(roc_auc, 3))

    # Save the trained model for the API
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

    # Save the training data with target and prediction for monitoring (Lesson 39 Demo 3)
    train_df = X_train.copy()
    train_df["target"] = y_train
    train_df["prediction"] = model.predict(X_train)
    train_df.to_csv("data/train_data.csv", index=False)

    # Save the test data with target and prediction for monitoring
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df["prediction"] = y_pred
    test_df.to_csv("data/reference_data.csv", index=False)

    # Confirm the monitoring files were saved
    print("Monitoring data saved to data/train_data.csv and data/reference_data.csv")
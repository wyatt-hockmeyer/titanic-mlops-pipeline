# monitor.py
# Builds Evidently AI reports for data drift and model performance, then checks alert thresholds
# Run from the project folder after train.py with: python monitor.py

# Import os to create the reports folder
import os

# Import pandas for working with tables of data
import pandas as pd

# Import joblib to load the trained model
import joblib

# Import the Evidently report tools used in Lesson 39 Demo 3
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Alert thresholds (documented exception: threshold check)
min_accuracy = 0.75
max_drift_share = 0.20

# Create the reports folder if it does not already exist
os.makedirs("reports", exist_ok=True)

# Load the training data (the baseline) and the test data (the "current" data)
reference = pd.read_csv("data/train_data.csv")
current = pd.read_csv("data/reference_data.csv")


# Define a function that builds one report, saves it, and returns its key results
def build_report(reference_data, current_data, file_name):
    # Create a report with the drift and classification presets (Lesson 39 Demo 3)
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    # Compare the current data against the reference data
    report.run(reference_data=reference_data, current_data=current_data)

    # Save the interactive report as an HTML file
    report.save_html("reports/" + file_name)
    print("Report saved : reports/" + file_name)

    # Turn the report results into a dictionary so the numbers can be checked
    results = report.as_dict()

    # The first metric holds the overall drift result
    drift = results["metrics"][0]["result"]

    # Find the metric that holds the accuracy of the current data
    accuracy = None
    for metric in results["metrics"]:
        if metric["metric"] == "ClassificationQualityMetric":
            accuracy = metric["result"]["current"]["accuracy"]

    # Return the share of drifted columns, the overall drift flag, and the accuracy
    return drift["share_of_drifted_columns"], drift["dataset_drift"], accuracy


# Define a function that prints the results and any alerts
def check_alerts(label, drift_share, dataset_drift, accuracy):
    # Show the results for this report
    print("\n" + label)
    print("Share of drifted columns :", round(drift_share, 3))
    print("Dataset drift detected :", dataset_drift)
    print("Accuracy :", round(accuracy, 3))

    # Track whether any alert fired
    alert = False

    # Alert if accuracy fell below the minimum
    if accuracy < min_accuracy:
        print("ALERT : accuracy", round(accuracy, 3), "is below the minimum of", min_accuracy, "- retraining recommended")
        alert = True

    # Alert if too many columns drifted
    if drift_share > max_drift_share:
        print("ALERT : drift share", round(drift_share, 3), "is above the maximum of", max_drift_share, "- review incoming data")
        alert = True

    # Confirm when no alert fired
    if alert == False:
        print("OK : no alerts")


# REPORT 1: training data compared with the test data (normal conditions)
drift_share, dataset_drift, accuracy = build_report(reference, current, "monitoring_report.html")
check_alerts("Report 1 : test data (normal conditions)", drift_share, dataset_drift, accuracy)

# REPORT 2: simulate a shift in incoming passengers to show the alerts working
# Copy the test data so the original is not changed
shifted = current.copy()

# Make the simulated passengers 20 years older
shifted["Age"] = shifted["Age"] + 20

# Make every simulated passenger travel in third class
shifted["Pclass"] = 3

# Remove the cabin records from every simulated passenger
shifted["HasCabin"] = 0

# Load the trained model
model = joblib.load("model.pkl")

# Separate the feature columns from the target and prediction columns
features = shifted.drop(columns=["target", "prediction"])

# Predict again, because the passengers' details changed
shifted["prediction"] = model.predict(features)

# Build the second report and check its alerts
drift_share, dataset_drift, accuracy = build_report(reference, shifted, "drift_simulation_report.html")
check_alerts("Report 2 : simulated shift in incoming passengers", drift_share, dataset_drift, accuracy)
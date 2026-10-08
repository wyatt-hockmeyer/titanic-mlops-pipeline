# preprocess.py
# Cleans the raw Titanic data, adds features, and saves a processed file
# Run from the project folder with: python preprocess.py

# Import pandas for working with tables of data
import pandas as pd

# Import numpy for numeric arrays
import numpy as np

# Import winsorize to cap extreme values (Data Wrangling lesson)
from scipy.stats.mstats import winsorize


# Define a function that groups each title into one of five categories
def group_title(title):
    # Keep the four common titles as they are
    if title == "Mr" or title == "Mrs" or title == "Miss" or title == "Master":
        return title
    # French and modern forms of Miss
    elif title == "Mlle" or title == "Ms":
        return "Miss"
    # French form of Mrs
    elif title == "Mme":
        return "Mrs"
    # Everything else is rare
    else:
        return "Rare"


# Define a function that cleans the data and adds features
def clean_data(df):
    # Fill blank ages with the median age
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print("Median age used :", age_median)

    # Fill blank ports with the most common port
    embarked_mode = df["Embarked"].value_counts().index[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print("Most common port used :", embarked_mode)

    # Create a flag for whether a cabin number was recorded
    df["HasCabin"] = df["Cabin"].notnull().astype(int)

    # Remove the original Cabin column
    df = df.drop(columns=["Cabin"])

    # Cap the highest 5% of fares
    df["Fare"] = np.array(winsorize(df["Fare"], limits=[0, 0.05]))

    # Create family size and travelling-alone features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Pull the title out of each name, then group it
    df["Title"] = df["Name"].apply(lambda name: name.split(", ")[1].split(".")[0])
    df["Title"] = df["Title"].apply(group_title)

    # Encode sex as a number: male 0, female 1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot encode port and title
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True, dtype="int")

    # Remove columns the model cannot use
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])

    # Return the cleaned data
    return df


# Location of the raw data file
raw_path = "data/titanic.csv"

# Location where the processed file will be saved
processed_path = "data/processed_titanic.csv"

# Track whether the raw file loaded
loaded = False

# Try to load the raw data
try:
    df = pd.read_csv(raw_path)
    loaded = True
except FileNotFoundError:
    print("Error : could not find", raw_path)

# Only continue if the file loaded
if loaded:
    # Show the size of the raw data
    print("Raw data shape :", df.shape)

    # Clean the data and add features
    df = clean_data(df)

    # Save the processed data without the row index
    df.to_csv(processed_path, index=False)

    # Confirm the result
    print("Processed data shape :", df.shape)
    print("Saved processed data to", processed_path)
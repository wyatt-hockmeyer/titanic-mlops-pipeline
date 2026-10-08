# test_model.py
# Checks the processed data and the trained model before the pipeline continues
# Run from the project folder with: python test_model.py
# A failed check stops the script with an AssertionError, which fails the GitHub Actions run

# Import pandas for working with tables of data
import pandas as pd

# Import joblib to load the saved model
import joblib

# Import the split tool and the two metrics the tests check
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score

# Minimum scores the model must reach to pass
min_recall = 0.70
min_roc_auc = 0.80

# The 16 columns preprocess.py must produce, in order
expected_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "HasCabin", "FamilySize", "IsAlone", "Embarked_Q", "Embarked_S", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare"]

# Load the processed data
df = pd.read_csv("data/processed_titanic.csv")

# TEST 1: the data has 891 rows and the expected 16 columns
assert df.shape[0] == 891, "Expected 891 rows"
assert df.columns.tolist() == expected_columns, "Columns do not match the expected list"
print("Test 1 passed : 891 rows and the expected 16 columns")

# TEST 2: no missing values remain after cleaning
assert df.isnull().sum().sum() == 0, "Processed data still has missing values"
print("Test 2 passed : no missing values")

# TEST 3: the target holds only 0 and 1
assert sorted(df["Survived"].unique().tolist()) == [0, 1], "Survived must contain only 0 and 1"
print("Test 3 passed : Survived contains only 0 and 1")

# Load the trained model saved by train.py
model = joblib.load("model.pkl")

# Separate the features (X) from the target (y)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Recreate the same test set train.py used
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predict classes and probabilities for the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# TEST 4: predictions are only 0 or 1
assert sorted(set(y_pred.tolist())) == [0, 1], "Predictions must be 0 or 1"
print("Test 4 passed : predictions are 0 or 1")

# Calculate the two scores the tests check
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# TEST 5: recall meets the minimum
assert recall >= min_recall, "Recall below minimum"
print("Test 5 passed : recall", round(recall, 3), "meets minimum", min_recall)

# TEST 6: ROC-AUC meets the minimum
assert roc_auc >= min_roc_auc, "ROC-AUC below minimum"
print("Test 6 passed : ROC-AUC", round(roc_auc, 3), "meets minimum", min_roc_auc)

# Confirm all tests passed
print("\nAll 6 tests passed")
# test_api.py
# Sends test requests to the running API and checks the answers
# Start the API first in another Terminal window, then run: python test_api.py

# Import os to read the API key from an environment variable
import os

# Import requests to send requests to the API (Lesson 39 Demo 1)
import requests

# Address of the prediction endpoint
url = "http://127.0.0.1:8000/predict/"

# Use the same API key the server uses
api_key = os.environ.get("API_KEY", "local-dev-key")

# Passenger 1: a woman in first class with a cabin
passenger_1 = {"Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C", "Title": "Mrs", "HasCabin": 1}

# Passenger 2: a man in third class travelling alone
passenger_2 = {"Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S", "Title": "Mr", "HasCabin": 0}

# TEST 1: passenger 1 with the correct key
response = requests.post(url, json=passenger_1, headers={"x-api-key": api_key})
print("Passenger 1 status :", response.status_code)
print("Passenger 1 result :", response.json())
assert response.status_code == 200, "Expected status 200"
assert response.json()["prediction"] == 1, "Expected passenger 1 to survive"

# TEST 2: passenger 2 with the correct key
response = requests.post(url, json=passenger_2, headers={"x-api-key": api_key})
print("\nPassenger 2 status :", response.status_code)
print("Passenger 2 result :", response.json())
assert response.status_code == 200, "Expected status 200"
assert response.json()["prediction"] == 0, "Expected passenger 2 not to survive"

# TEST 3: passenger 1 with a wrong key must be rejected
response = requests.post(url, json=passenger_1, headers={"x-api-key": "wrong-key"})
print("\nWrong key status :", response.status_code)
print("Wrong key result :", response.json())
assert response.status_code == 401, "Expected status 401 for a wrong key"

# TEST 4: passenger 1 with no key must be rejected
response = requests.post(url, json=passenger_1)
print("\nNo key status :", response.status_code)
print("No key result :", response.json())
assert response.status_code == 401, "Expected status 401 for no key"

# Confirm all tests passed
print("\nAll 4 API tests passed")
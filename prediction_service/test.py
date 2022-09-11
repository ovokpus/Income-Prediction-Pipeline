import requests

data_point = {
    "age": "45",
    "workclass": "Private",
    "financialWeight": 226717,
    "education": "Bachelors",
    "educationNum": 13,
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Female",
    "capitalGain": 0,
    "capitalLoss": 0,
    "hoursPerWeek": 40,
    "nativeCountry": "United-States",
}

url = "http://localhost:9696/predict"
response = requests.post(url, json=data_point)
print(response.json())

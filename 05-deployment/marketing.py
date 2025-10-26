import requests

# url = "http://localhost:9696/predict"
url = "https://summer-sunset-6221.fly.dev/predict"
customer = {
    "gender": "male",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "mailed_check",
    "tenure": 6,
    "monthlycharges": 23.85,
    "totalcharges": 263.55
}

response = requests.post(url, json=customer)
churn  = response.json()

print(f"Probability of churning = {churn['churn_probability']}")
print(f"Churn = {churn['churn']}")

if churn['churn_probability'] >= 0.5:
    print("Send email with promo")
else:
    print("Don't do anything")
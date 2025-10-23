
import pickle


with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


customer = {
  'gender': 'male',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'yes',
  'deviceprotection': 'no',
  'techsupport': 'no',
  'streamingtv': 'no',
  'streamingmovies': 'no',
  'contract': 'month-to-month',
  'paperlessbilling': 'yes',
  'paymentmethod': 'mailed_check',
  'tenure': 6,
  'monthlycharges': 23.85,
  'totalcharges': 263.55
}

churn = pipeline.predict_proba(customer)[0, 1]
print("Probability of churn = ", churn)

if churn >= 0.5:
    print("Send email with promo")
else:
    print("Don't do anything")

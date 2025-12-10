import requests

url = "https://ewvhdmq9x9.execute-api.ap-southeast-2.amazonaws.com/test/predict"

request = {"url": "http://bit.ly/mlbookcamp-pants"}

result = requests.post(url, json=request).json()
print(result)
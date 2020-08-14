import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'mood':'Happy', 'step_count':6543, 'calories_burned':234,'hours_sleep':8, 'weight_kg':71})

print(r.json())

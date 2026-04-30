import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

hours = np.random.randint(0, 24, n)

amounts = []
for h in hours:
    if 2 <= h <= 5:  # Late night = higher fraud chance
        amounts.append(np.random.choice([
            np.random.uniform(8000, 50000),
            np.random.uniform(100, 2000)
        ], p=[0.7, 0.3]))
    else:
        amounts.append(np.random.uniform(100, 20000))

amounts = np.array(amounts)

merchants = np.random.choice(
    ['grocery', 'electronics', 'food', 'travel', 'unknown', 'crypto'],
    n, p=[0.30, 0.20, 0.25, 0.10, 0.10, 0.05]
)

locations = np.random.choice(
    ['home_city', 'nearby_city', 'foreign', 'unknown'],
    n, p=[0.60, 0.25, 0.10, 0.05]
)

new_device = np.random.choice([0, 1], n, p=[0.85, 0.15])
failed_attempts = np.random.randint(0, 6, n)

# Fraud logic — realistic rules
fraud = []
for i in range(n):
    score = 0
    if amounts[i] > 15000:
        score += 2
    if 2 <= hours[i] <= 5:
        score += 2
    if merchants[i] in ['unknown', 'crypto']:
        score += 2
    if locations[i] in ['foreign', 'unknown']:
        score += 2
    if new_device[i] == 1:
        score += 1
    if failed_attempts[i] >= 3:
        score += 2

    fraud.append(1 if score >= 5 else 0)

df = pd.DataFrame({
    'amount': amounts,
    'hour': hours,
    'merchant_type': merchants,
    'location': locations,
    'new_device': new_device,
    'failed_attempts': failed_attempts,
    'is_fraud': fraud
})

df.to_csv('transactions.csv', index=False)
print(f"Dataset ready! Total: {n} | Fraud: {sum(fraud)} | Legit: {n - sum(fraud)}")
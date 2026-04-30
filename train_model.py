import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import shap
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('transactions.csv')

# Convert text columns to numbers (ML sirf numbers samajhta hai)
le_merchant = LabelEncoder()
le_location = LabelEncoder()

df['merchant_type'] = le_merchant.fit_transform(df['merchant_type'])
df['location'] = le_location.fit_transform(df['location'])

# Save encoders (app.py mein bhi kaam aayenge)
joblib.dump(le_merchant, 'le_merchant.pkl')
joblib.dump(le_location, 'le_location.pkl')

# Features aur target alag karo
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Train/Test split — 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model banao aur train karo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Model ki accuracy check karo
y_pred = model.predict(X_test)
print("=== Model Performance ===")
print(classification_report(y_test, y_pred))

# Model save karo
joblib.dump(model, 'fraud_model.pkl')
print("\n✅ Model saved!")

# SHAP — model explain karo (hatke factor)
print("\nGenerating SHAP explanation chart...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

plt.figure()
shap.summary_plot(
    shap_values[:,:,1] if isinstance(shap_values, list) == False else shap_values[1],
    X_test[:100],
    show=False
)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
print("✅ SHAP chart saved as shap_summary.png")
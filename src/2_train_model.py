import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_FILE  = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures.csv')
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'gesture_model.pkl')

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
print(f"  Rows: {len(df)}")
print(f"  Gestures: {sorted(df['label'].unique())}")

X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\nTraining on {len(X_train)} samples...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nPer-gesture report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': model, 'encoder': le}, f)

print(f"\n💾 Model saved to models/gesture_model.pkl")
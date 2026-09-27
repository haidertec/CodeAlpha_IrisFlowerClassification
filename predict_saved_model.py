"""
Saved Decision Tree model se Iris prediction
Bina dobara train kiye!
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import load_iris

# ---------- Feature engineering (same as training) ----------
def engineer_features_v2(X):
    sl, sw, pl, pw = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    eps = 1e-6
    feats = [sl, sw, pl, pw]
    feats += [pl/pw, sl/sw, pl/sl, pw/sw, (pl*pw)/(sl*sw + eps)]
    feats += [pl*pw, sl*sw, np.sqrt(pl**2 + pw**2), np.sqrt(sl**2 + sw**2)]
    feats += [pl**2, pw**2, sl**2, sw**2]
    feats += [pl**3, pw**3]
    feats += [np.log1p(pl), np.log1p(pw), np.log1p(sl), np.log1p(sw)]
    feats += [pl*sl, pl*sw, pw*sl, pw*sw]
    feats += [pl - pw, sl - sw, pl - sl, pw - sw]
    feats += [pl + pw, sl + sw]
    feats += [(pl/pw) / (sl/sw + eps), (pl*pw) / (sl + sw + eps)]
    return np.column_stack(feats)

# ---------- Load saved model ----------
BASE = Path("iris_ultra")
MODELS = BASE / "models"

latest_model = sorted(MODELS.glob("DT_*.pkl"))[-1]
latest_scaler = sorted(MODELS.glob("scaler_*.pkl"))[-1]

model = joblib.load(latest_model)
scaler = joblib.load(latest_scaler)

print(f"✅ Loaded model : {latest_model.name}")
print(f"✅ Loaded scaler: {latest_scaler.name}")

# ---------- Prediction function ----------
iris = load_iris()
class_names = iris.target_names

def predict(sepal_l, sepal_w, petal_l, petal_w):
    """Raw measurements se species predict karein."""
    raw = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    eng = engineer_features_v2(raw)
    eng_s = scaler.transform(eng)
    pred = model.predict(eng_s)[0]
    proba = model.predict_proba(eng_s)[0]
    return class_names[pred], proba

# ---------- Test predictions ----------
print("\n" + "=" * 60)
print("Demo Predictions")
print("=" * 60)

samples = [
    (5.1, 3.5, 1.4, 0.2, "setosa"),
    (6.0, 2.7, 5.1, 1.6, "virginica"),
    (5.9, 3.0, 4.2, 1.5, "versicolor"),
]

for sl, sw, pl, pw, expected in samples:
    species, proba = predict(sl, sw, pl, pw)
    status = "✅" if species == expected else "❌"
    print(f"\n{status} Input: [{sl}, {sw}, {pl}, {pw}]")
    print(f"   Predicted: {species} (expected: {expected})")
    for name, p in zip(class_names, proba):
        bar = "█" * int(p * 30)
        print(f"   {name:12s}: {p*100:5.2f}%  {bar}")

# ---------- Interactive mode ----------
print("\n" + "=" * 60)
print("Interactive Mode (type 'q' to quit)")
print("=" * 60)

while True:
    try:
        inp = input("\n4 measurements (sl sw pl pw) daalein, ya 'q': ").strip()
        if inp.lower() == 'q':
            break
        parts = inp.split()
        if len(parts) != 4:
            print("❌ Zaroori: 4 numbers (sepal_l sepal_w petal_l petal_w)")
            continue
        sl, sw, pl, pw = map(float, parts)
        species, proba = predict(sl, sw, pl, pw)
        print(f"\n   🎯 Predicted: {species.upper()}")
        for name, p in zip(class_names, proba):
            bar = "█" * int(p * 40)
            print(f"   {name:12s}: {p*100:6.2f}%  {bar}")
    except KeyboardInterrupt:
        print("\n👋 Bye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
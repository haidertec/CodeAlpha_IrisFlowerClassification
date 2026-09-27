"""
=========================================================
  IRIS CLASSIFICATION - ULTRA ADVANCED (FAST + FIXED)
=========================================================
  Speed optimizations:
    - Reduced AutoML search space (25 configs)
    - Fewer GA generations (5)
    - Fewer Pareto trials (15)
    - Smaller deep ensemble (3 models)
    - Faster SHAP (fewer trees)
  Bug fixes:
    - PDP multi-class target specified
    - GA gamma type casting
    - liblinear removed
=========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, time, hashlib, logging, random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap
import joblib
from scipy import stats

# =========================================================
# CONFIG
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = Path("iris_ultra")
OUTPUT = BASE / "outputs"
MODELS = BASE / "models"
REGISTRY = BASE / "registry"
EXPERIMENTS = BASE / "experiments"

for d in [BASE, OUTPUT, MODELS, REGISTRY, EXPERIMENTS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT / "run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
COLORS = ["#2ecc71", "#e67e22", "#3498db"]

def header(t):
    print("\n" + "╔" + "═" * 75 + "╗")
    print(f"║  {t:<72s}║")
    print("╚" + "═" * 75 + "╝")

def save_fig(name):
    plt.savefig(OUTPUT / name, bbox_inches="tight", dpi=100)

# =========================================================
# EXPERIMENT TRACKER
# =========================================================
@dataclass
class Experiment:
    run_id: str
    model_name: str
    params: Dict
    metrics: Dict
    timestamp: str
    duration_sec: float

class ExperimentTracker:
    def __init__(self, root: Path):
        self.root = root
        self.runs_file = root / "runs.json"
        self.runs = json.loads(self.runs_file.read_text()) if self.runs_file.exists() else []

    def log(self, exp: Experiment):
        self.runs.append(asdict(exp))
        self.runs_file.write_text(json.dumps(self.runs, indent=2))

tracker = ExperimentTracker(EXPERIMENTS)

# =========================================================
# STEP 1: Load Data
# =========================================================
header("STEP 1: Loading Data")
iris = load_iris()
X_raw, y = iris.data, iris.target
class_names = iris.target_names
print(f"  Samples  : {X_raw.shape[0]}")
print(f"  Features : {X_raw.shape[1]}")
print(f"  Classes  : {[str(c) for c in class_names]}")

# =========================================================
# STEP 2: Advanced Feature Engineering
# =========================================================
header("STEP 2: Advanced Feature Engineering (35 features)")

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

X = engineer_features_v2(X_raw)
feat_names = [
    "sl", "sw", "pl", "pw",
    "pl_pw_ratio", "sl_sw_ratio", "pl_sl_ratio", "pw_sw_ratio", "area_ratio",
    "petal_area", "sepal_area", "petal_diag", "sepal_diag",
    "pl_sq", "pw_sq", "sl_sq", "sw_sq",
    "pl_cu", "pw_cu",
    "log_pl", "log_pw", "log_sl", "log_sw",
    "pl_sl", "pl_sw", "pw_sl", "pw_sw",
    "pl_minus_pw", "sl_minus_sw", "pl_minus_sl", "pw_minus_sw",
    "pl_plus_pw", "sl_plus_sw",
    "ratio_of_ratios", "area_ratio_2",
]

print(f"  Original : 4 features")
print(f"  Total    : {X.shape[1]} features")

# =========================================================
# STEP 3: Advanced Data Splitting
# =========================================================
header("STEP 3: Advanced Data Splitting")

X_tv, X_te, y_tv, y_te = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tv, y_tv, test_size=0.176, random_state=SEED, stratify=y_tv
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

print(f"  Train : {X_tr.shape}")
print(f"  Val   : {X_va.shape}")
print(f"  Test  : {X_te.shape}")

# =========================================================
# STEP 4: PCA + Kernel PCA
# =========================================================
header("STEP 4: PCA + Kernel PCA")

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_tr_s)
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5, random_state=SEED)
X_kpca = kpca.fit_transform(X_tr_s)

print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (data, title) in zip(axes, [(X_pca, "PCA"), (X_kpca, "Kernel PCA (RBF)")]):
    for i, (cls, c) in enumerate(zip(class_names, COLORS)):
        mask = y_tr == i
        ax.scatter(data[mask, 0], data[mask, 1], c=c, label=cls,
                   s=80, alpha=0.7, edgecolors="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
save_fig("01_pca_kpca.png")
plt.show()

# =========================================================
# STEP 5: AutoML - FAST (25 configs)
# =========================================================
header("STEP 5: AutoML - Automated Model Search (Fast)")

def get_model_space():
    spaces = []
    # Logistic Regression (3 configs)
    for C in [0.1, 1, 10]:
        spaces.append((f"LR_C{C}",
                       LogisticRegression(C=C, max_iter=1000, random_state=SEED)))
    # SVM (6 configs)
    for C in [0.1, 1, 10]:
        for kernel in ["rbf", "linear"]:
            spaces.append((f"SVM_{kernel}_C{C}",
                           SVC(C=C, kernel=kernel, probability=True, random_state=SEED)))
    # KNN (3 configs)
    for k in [3, 5, 7]:
        spaces.append((f"KNN_k{k}", KNeighborsClassifier(n_neighbors=k)))
    # Random Forest (3 configs)
    for n in [50, 100, 200]:
        spaces.append((f"RF_n{n}", RandomForestClassifier(n_estimators=n, random_state=SEED)))
    # Others (5 configs)
    spaces.append(("GB", GradientBoostingClassifier(random_state=SEED)))
    spaces.append(("HGB", HistGradientBoostingClassifier(random_state=SEED)))
    spaces.append(("ET", ExtraTreesClassifier(n_estimators=100, random_state=SEED)))
    spaces.append(("DT", DecisionTreeClassifier(random_state=SEED)))
    spaces.append(("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED)))
    return spaces

model_space = get_model_space()
print(f"  Searching {len(model_space)} model configurations...")

automl_results = []
for name, model in model_space:
    try:
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        val_acc = accuracy_score(y_va, model.predict(X_va_s))
        val_f1 = f1_score(y_va, model.predict(X_va_s), average="weighted")
        dt = time.time() - t0
        automl_results.append({"name": name, "model": model,
                                "val_acc": val_acc, "val_f1": val_f1, "time": dt})
    except Exception as e:
        log.warning(f"Model {name} failed: {e}")

automl_results.sort(key=lambda x: (-x["val_acc"], -x["val_f1"], x["time"]))

print(f"\n  Top 10 Models:")
print(f"  {'Rank':<5s} {'Model':<25s} {'Val Acc':>10s} {'Val F1':>10s} {'Time(s)':>10s}")
print("  " + "-" * 65)
for rank, r in enumerate(automl_results[:10], 1):
    print(f"  {rank:<5d} {r['name']:<25s} {r['val_acc']:>10.4f} "
          f"{r['val_f1']:>10.4f} {r['time']:>10.3f}")

# =========================================================
# STEP 6: Genetic Algorithm (FAST - 5 gens)
# =========================================================
header("STEP 6: Genetic Algorithm (Fast)")

class GeneticOptimizer:
    def __init__(self, param_space, fitness_fn, population_size=10,
                 generations=5, mutation_rate=0.2, seed=42):
        self.param_space = param_space
        self.fitness_fn = fitness_fn
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = np.random.RandomState(seed)
        self.history = []

    def _random_individual(self):
        return {k: self.rng.choice(v) for k, v in self.param_space.items()}

    def _mutate(self, individual):
        new = individual.copy()
        for k in new:
            if self.rng.rand() < self.mutation_rate:
                new[k] = self.rng.choice(self.param_space[k])
        return new

    def _crossover(self, p1, p2):
        return {k: p1[k] if self.rng.rand() < 0.5 else p2[k] for k in p1}

    def run(self):
        population = [self._random_individual() for _ in range(self.pop_size)]
        best_overall = None
        best_score = -np.inf

        for gen in range(self.generations):
            fitness = []
            for ind in population:
                try:
                    fitness.append(self.fitness_fn(ind))
                except Exception:
                    fitness.append(0.0)
            fitness = np.array(fitness)

            best_idx = int(np.argmax(fitness))
            if fitness[best_idx] > best_score:
                best_score = fitness[best_idx]
                best_overall = population[best_idx].copy()

            self.history.append({"gen": gen, "best_score": float(best_score),
                                  "mean_score": float(np.mean(fitness))})
            print(f"  Gen {gen+1:2d}/{self.generations}: "
                  f"best={best_score:.4f}, mean={np.mean(fitness):.4f}")

            selected = []
            for _ in range(self.pop_size):
                i, j = self.rng.choice(self.pop_size, 2, replace=False)
                selected.append(population[i] if fitness[i] > fitness[j] else population[j])

            new_pop = [best_overall.copy()]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.rng.choice(len(selected), 2, replace=False)
                child = self._crossover(selected[p1], selected[p2])
                child = self._mutate(child)
                new_pop.append(child)
            population = new_pop

        return best_overall, best_score


ga_space = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", 0.01, 0.1, 1.0],
    "kernel": ["rbf", "linear"],
}

def svm_fitness(params):
    p = dict(params)
    p["C"] = float(p["C"])
    g = p["gamma"]
    if isinstance(g, (str, np.str_)):
        g_str = str(g)
        p["gamma"] = g_str if g_str in ("scale", "auto") else float(g_str)
    else:
        p["gamma"] = float(g)
    p["kernel"] = str(p["kernel"])
    m = SVC(**p, probability=True, random_state=SEED)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)   # 3-fold for speed
    return cross_val_score(m, X_tr_s, y_tr, cv=cv, n_jobs=-1).mean()


ga = GeneticOptimizer(ga_space, svm_fitness,
                       population_size=10, generations=5, seed=SEED)
ga_best, ga_score = ga.run()
print(f"\n  GA Best params : {ga_best}")
print(f"  GA Best score  : {ga_score:.4f}")

# GA plot
ga_hist_df = pd.DataFrame(ga.history)
plt.figure(figsize=(10, 5))
plt.plot(ga_hist_df["gen"] + 1, ga_hist_df["best_score"],
         "o-", color="#2ecc71", label="Best", lw=2)
plt.plot(ga_hist_df["gen"] + 1, ga_hist_df["mean_score"],
         "s-", color="#e74c3c", label="Mean", lw=2)
plt.xlabel("Generation")
plt.ylabel("Fitness (CV Accuracy)")
plt.title("Genetic Algorithm Convergence", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("02_ga_convergence.png")
plt.show()

# =========================================================
# STEP 7: Multi-Objective (FAST - 15 trials)
# =========================================================
header("STEP 7: Multi-Objective (Accuracy vs Speed)")

def pareto_objective(trial):
    C = trial.suggest_float("C", 0.01, 100, log=True)
    gamma = trial.suggest_float("gamma", 0.001, 10, log=True)
    m = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=SEED)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    t0 = time.time()
    try:
        score = cross_val_score(m, X_tr_s, y_tr, cv=cv, n_jobs=-1).mean()
    except Exception:
        score = 0.0
    return score, time.time() - t0

study_mo = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(seed=SEED)
)
study_mo.optimize(pareto_objective, n_trials=15, show_progress_bar=False)

pareto_trials = study_mo.best_trials
print(f"  Pareto front size: {len(pareto_trials)}")
for t in pareto_trials[:5]:
    print(f"    Trial {t.number}: acc={t.values[0]:.4f}, time={t.values[1]:.3f}s")

# =========================================================
# STEP 8: Ensemble Methods
# =========================================================
header("STEP 8: Ensemble Methods")

ensemble_scores = {}

voting = VotingClassifier([
    ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=SEED)),
], voting="soft")
voting.fit(X_tr_s, y_tr)
ensemble_scores["Voting"] = accuracy_score(y_va, voting.predict(X_va_s))

stacking = StackingClassifier([
    ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=SEED)),
], final_estimator=LogisticRegression(max_iter=1000, random_state=SEED), cv=3)
stacking.fit(X_tr_s, y_tr)
ensemble_scores["Stacking"] = accuracy_score(y_va, stacking.predict(X_va_s))

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=SEED),
    n_estimators=50, random_state=SEED
)
bagging.fit(X_tr_s, y_tr)
ensemble_scores["Bagging"] = accuracy_score(y_va, bagging.predict(X_va_s))

for name, acc in ensemble_scores.items():
    print(f"  {name:15s} -> {acc:.4f}")

# =========================================================
# STEP 9: Deep Ensemble (FAST - 3 models)
# =========================================================
header("STEP 9: Deep Ensemble (Fast)")

class DeepEnsemble:
    def __init__(self, n_models=3, hidden=(64, 32), seed=42):
        self.n_models = n_models
        self.hidden = hidden
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        for i in range(self.n_models):
            nn = MLPClassifier(
                hidden_layer_sizes=self.hidden,
                activation="relu", solver="adam",
                alpha=1e-3, batch_size=16,
                max_iter=500, early_stopping=True,
                validation_fraction=0.2, n_iter_no_change=20,
                random_state=self.seed + i, verbose=False
            )
            nn.fit(X, y)
            self.models.append(nn)

    def predict_proba(self, X):
        return np.array([m.predict_proba(X) for m in self.models]).mean(axis=0)

    def predict_with_uncertainty(self, X):
        preds = np.array([m.predict_proba(X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

ensemble = DeepEnsemble(n_models=3, seed=SEED)
ensemble.fit(X_tr_s, y_tr)

ens_mean, ens_std = ensemble.predict_with_uncertainty(X_te_s)
ens_pred = ens_mean.argmax(axis=1)
ens_acc = accuracy_score(y_te, ens_pred)

print(f"  Deep Ensemble ({ensemble.n_models} models) test acc: {ens_acc:.4f}")
print(f"  Mean uncertainty: {ens_std.mean():.4f}")

uncertainty = ens_std.max(axis=1)
top_uncertain = np.argsort(-uncertainty)[:5]
print("\n  Top 5 uncertain samples:")
for idx in top_uncertain:
    print(f"    Sample {idx}: true={class_names[y_te[idx]]:<12s} "
          f"pred={class_names[ens_pred[idx]]:<12s} unc={uncertainty[idx]:.4f}")

# =========================================================
# STEP 10: Adversarial Robustness
# =========================================================
header("STEP 10: Adversarial Robustness")

best_trad = automl_results[0]["model"]
best_trad_name = automl_results[0]["name"]
base_acc = accuracy_score(y_te, best_trad.predict(X_te_s))

def random_perturbation_attack(model, X, y, epsilon=0.1):
    X_adv = X.copy()
    for i in range(len(X)):
        noise = np.random.randn(X.shape[1]) * epsilon
        X_perturbed = X[i] + noise
        if model.predict(X_perturbed.reshape(1, -1))[0] != y[i]:
            X_adv[i] = X_perturbed
    return X_adv

print(f"  Testing {best_trad_name} (base acc: {base_acc:.4f})")
for eps in [0.05, 0.1, 0.2, 0.3]:
    X_adv = random_perturbation_attack(best_trad, X_te_s, y_te, epsilon=eps)
    acc_adv = accuracy_score(y_te, best_trad.predict(X_adv))
    print(f"    ε = {eps:.2f} -> acc: {acc_adv:.4f} (drop: {base_acc - acc_adv:+.4f})")

# =========================================================
# STEP 11: Conformal Prediction
# =========================================================
header("STEP 11: Conformal Prediction")

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None

    def calibrate(self, model, X_cal, y_cal):
        proba = model.predict_proba(X_cal)
        scores = 1 - proba[np.arange(len(y_cal)), y_cal]
        n = len(scores)
        self.q = np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n)

    def predict_set(self, model, X):
        proba = model.predict_proba(X)
        return [np.where((1 - proba[i]) <= self.q)[0] for i in range(len(X))]

cp = ConformalPredictor(alpha=0.1)
cp.calibrate(best_trad, X_va_s, y_va)
pred_sets = cp.predict_set(best_trad, X_te_s)

avg_size = np.mean([len(s) for s in pred_sets])
coverage = np.mean([y_te[i] in pred_sets[i] for i in range(len(y_te))])

print(f"  Target coverage : 90%")
print(f"  Actual coverage : {coverage*100:.1f}%")
print(f"  Avg set size    : {avg_size:.2f} classes")

# =========================================================
# STEP 12: Fairness Analysis
# =========================================================
header("STEP 12: Fairness Analysis")

sepal_width_te = X_te[:, 1]
median_sw = np.median(X_te[:, 1])
groups = (sepal_width_te > median_sw).astype(int)
y_pred = best_trad.predict(X_te_s)

print(f"  Group 0 (small sepal width): n={sum(groups == 0)}")
print(f"  Group 1 (large sepal width): n={sum(groups == 1)}")

for g in [0, 1]:
    mask = groups == g
    acc = accuracy_score(y_te[mask], y_pred[mask])
    print(f"  Group {g} accuracy: {acc:.4f}")

pred_rate_g0 = np.mean(y_pred[groups == 0] == 2)
pred_rate_g1 = np.mean(y_pred[groups == 1] == 2)
dp_diff = abs(pred_rate_g0 - pred_rate_g1)
print(f"  Demographic parity diff: {dp_diff:.4f} (fair if < 0.1)")

# =========================================================
# STEP 13: Data Drift Detection
# =========================================================
header("STEP 13: Data Drift Detection")

def detect_drift(X_ref, X_new, feature_names, threshold=0.05):
    results = []
    for i, fname in enumerate(feature_names):
        stat, p = stats.ks_2samp(X_ref[:, i], X_new[:, i])
        results.append({"feature": fname, "ks_stat": stat, "p_value": p,
                         "drift": p < threshold})
    return pd.DataFrame(results)

drift_df = detect_drift(X_tr_s, X_te_s, feat_names)
drifted = drift_df[drift_df["drift"]]

print(f"  Features with drift: {len(drifted)}/{len(feat_names)}")
if len(drifted) > 0:
    print(drifted.head(10).to_string(index=False))
else:
    print("  ✅ No significant drift detected")

# =========================================================
# STEP 14: Explainability Trio
# =========================================================
header("STEP 14: Explainability Trio")

# 14.1 SHAP (FAST - 50 trees)
print("  [1/3] SHAP...")
rf = RandomForestClassifier(n_estimators=50, random_state=SEED)
rf.fit(X_tr_s, y_tr)
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X_te_s)

if isinstance(shap_vals, list):
    shap_plot = shap_vals
elif len(shap_vals.shape) == 3:
    shap_plot = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
else:
    shap_plot = shap_vals

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_plot, X_te_s, feature_names=feat_names,
                  class_names=class_names, show=False)
plt.tight_layout()
save_fig("14_shap.png")
plt.show()

# 14.2 Permutation Importance (FAST - 5 repeats)
print("  [2/3] Permutation importance...")
perm = permutation_importance(best_trad, X_te_s, y_te,
                                n_repeats=5, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": feat_names,
    "importance": perm.importances_mean,
    "std": perm.importances_std
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(perm_df["feature"].head(15)[::-1],
         perm_df["importance"].head(15)[::-1],
         xerr=perm_df["std"].head(15)[::-1], color="#9b59b6")
plt.xlabel("Mean accuracy decrease")
plt.title("Top 15 Permutation Importances", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("15_permutation.png")
plt.show()

print("\n  Top 5 features:")
for _, row in perm_df.head(5).iterrows():
    print(f"    {row['feature']:20s}: {row['importance']:.4f} ± {row['std']:.4f}")

# 14.3 Partial Dependence (FIXED for multi-class)
print("  [3/3] Partial dependence plots...")
top_4 = perm_df["feature"].head(4).tolist()
top_4_idx = [feat_names.index(f) for f in top_4]

fig, axes = plt.subplots(len(top_4_idx), 3, figsize=(18, 3.5 * len(top_4_idx)))
if len(top_4_idx) == 1:
    axes = axes.reshape(1, -1)

for row, (idx, name) in enumerate(zip(top_4_idx, top_4)):
    for col, class_id in enumerate([0, 1, 2]):
        ax = axes[row, col]
        try:
            PartialDependenceDisplay.from_estimator(
                best_trad, X_te_s, [idx],
                target=class_id,           # ← FIX for multi-class
                ax=ax,
                feature_names=feat_names
            )
            ax.set_title(f"PDP: {name} | {class_names[class_id]}",
                         fontsize=9, fontweight="bold")
        except Exception as e:
            ax.text(0.5, 0.5, f"PDP failed:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
save_fig("16_pdp.png")
plt.show()

# =========================================================
# STEP 15: CV Strategies
# =========================================================
header("STEP 15: CV Strategies Comparison")

strategies = {
    "Stratified 5-Fold":  StratifiedKFold(5, shuffle=True, random_state=SEED),
    "Stratified 10-Fold": StratifiedKFold(10, shuffle=True, random_state=SEED),
    "Repeated 5x3":       RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED),
}

cv_results = []
for name, cv in strategies.items():
    cv_res = cross_validate(best_trad, X, y, cv=cv,
                             scoring=["accuracy", "f1_weighted"],
                             n_jobs=-1)
    cv_results.append({
        "strategy": name,
        "acc_mean": cv_res["test_accuracy"].mean(),
        "acc_std":  cv_res["test_accuracy"].std(),
        "f1_mean":  cv_res["test_f1_weighted"].mean(),
        "fit_time": cv_res["fit_time"].mean(),
    })

cv_df = pd.DataFrame(cv_results)
print(cv_df.to_string(index=False))

# =========================================================
# STEP 16: Hyperparameter Sensitivity
# =========================================================
header("STEP 16: Hyperparameter Sensitivity (SVM)")

C_range = [0.1, 1, 10]
gamma_range = ["scale", 0.01, 0.1, 1.0]

heatmap_data = np.zeros((len(C_range), len(gamma_range)))
for i, C in enumerate(C_range):
    for j, g in enumerate(gamma_range):
        try:
            m = SVC(C=C, gamma=g, kernel="rbf", probability=True, random_state=SEED)
            cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
            heatmap_data[i, j] = cross_val_score(m, X_tr_s, y_tr, cv=cv, n_jobs=-1).mean()
        except Exception:
            heatmap_data[i, j] = np.nan

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=[str(g) for g in gamma_range],
            yticklabels=[str(c) for c in C_range])
plt.xlabel("gamma", fontweight="bold")
plt.ylabel("C", fontweight="bold")
plt.title("SVM Hyperparameter Sensitivity", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("17_hyperparam_heatmap.png")
plt.show()

# =========================================================
# STEP 17: Model Registry
# =========================================================
header("STEP 17: Model Registry")

version = datetime.now().strftime("v%Y%m%d_%H%M%S")
model_path = MODELS / f"{best_trad_name}_{version}.pkl"
joblib.dump(best_trad, model_path)
joblib.dump(scaler, MODELS / f"scaler_{version}.pkl")

registry_entry = {
    "version": version,
    "model_name": best_trad_name,
    "model_path": str(model_path),
    "accuracy": float(accuracy_score(y_te, best_trad.predict(X_te_s))),
    "features_count": len(feat_names),
    "trained_at": datetime.now().isoformat(),
    "stage": "production"
}

registry_file = REGISTRY / "registry.json"
registry = json.loads(registry_file.read_text()) if registry_file.exists() else []
registry.append(registry_entry)
registry_file.write_text(json.dumps(registry, indent=2))

print(f"  ✅ Registered version: {version}")
print(f"  📁 Model: {model_path}")

# =========================================================
# STEP 18: Final Evaluation
# =========================================================
header("STEP 18: Final Test Set Evaluation")

final_acc = accuracy_score(y_te, best_trad.predict(X_te_s))
final_f1 = f1_score(y_te, best_trad.predict(X_te_s), average="weighted")

print(f"  Best Model    : {best_trad_name}")
print(f"  Test Accuracy : {final_acc*100:.2f}%")
print(f"  Test F1       : {final_f1:.4f}")
print(f"  Deep Ensemble : {ens_acc*100:.2f}%")

print("\n  Classification Report:")
print(classification_report(y_te, best_trad.predict(X_te_s),
                             target_names=class_names))

# =========================================================
# STEP 19: Final Summary
# =========================================================
header("🎉 ULTRA ADVANCED TASK COMPLETED (FAST)")

print(f"""
  ═══════════════════════════════════════════════════════════════
    IRIS CLASSIFICATION - ULTRA ADVANCED (FAST)
  ═══════════════════════════════════════════════════════════════

  Dataset           : 150 samples, {X.shape[1]} features, 3 classes
  Models Searched   : {len(model_space)} configurations
  Best Model        : {best_trad_name}
  Test Accuracy     : {final_acc*100:.2f}%
  Deep Ensemble     : {ens_acc*100:.2f}%

  ✅ AutoML (fast)
  ✅ Genetic Algorithm
  ✅ Multi-Objective (Pareto)
  ✅ Deep Ensemble
  ✅ Adversarial Robustness
  ✅ Conformal Prediction
  ✅ Fairness Analysis
  ✅ Data Drift Detection
  ✅ SHAP + Permutation + PDP
  ✅ Hyperparameter Sensitivity
  ✅ Model Registry

  Output: {OUTPUT}/
""")

log.info("FAST TASK COMPLETED SUCCESSFULLY")
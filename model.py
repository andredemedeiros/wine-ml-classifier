import random
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"]).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=SEED, stratify=y
)

# ── Preprocessing steps ───────────────────────────────────────────────────────
preprocessors = {
    "StandardScaler":  StandardScaler(),
    "MinMaxScaler":    MinMaxScaler(),
    "RobustScaler":    RobustScaler(),
    "Normalizer":      Normalizer(),
    "PCA(2)":          Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=2, random_state=SEED))]),
    "PCA(3)":          Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=3, random_state=SEED))]),
    "SelectKBest(2)":  Pipeline([("sc", StandardScaler()), ("skb", SelectKBest(f_classif, k=2))]),
    "SelectKBest(3)":  Pipeline([("sc", StandardScaler()), ("skb", SelectKBest(f_classif, k=3))]),
    "None":            "passthrough",
}

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "LogisticRegression":       LogisticRegression(max_iter=1000, random_state=SEED),
    "RidgeClassifier":          RidgeClassifier(),
    "SVM(rbf)":                 SVC(kernel="rbf",    random_state=SEED),
    "SVM(linear)":              SVC(kernel="linear", random_state=SEED),
    "SVM(poly)":                SVC(kernel="poly",   random_state=SEED),
    "KNN(3)":                   KNeighborsClassifier(n_neighbors=3),
    "KNN(5)":                   KNeighborsClassifier(n_neighbors=5),
    "KNN(7)":                   KNeighborsClassifier(n_neighbors=7),
    "RandomForest(100)":        RandomForestClassifier(n_estimators=100, random_state=SEED),
    "RandomForest(200)":        RandomForestClassifier(n_estimators=200, random_state=SEED),
    "ExtraTrees":               ExtraTreesClassifier(n_estimators=100,  random_state=SEED),
    "GradientBoosting":         GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "GradientBoosting(lr0.05)": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED),
    "NaiveBayes":               GaussianNB(),
    "LDA":                      LinearDiscriminantAnalysis(),
}

# ── Grid search ───────────────────────────────────────────────────────────────
results = []

for (prep_name, prep), (model_name, model) in product(preprocessors.items(), models.items()):
    try:
        pipe = Pipeline([("prep", prep), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        pipe.fit(X_train, y_train)
        pred      = pipe.predict(X_test)
        test_acc  = accuracy_score(y_test, pred)
        test_f1   = f1_score(y_test, pred, average="weighted")
        results.append({
            "Preprocessor":   prep_name,
            "Model":          model_name,
            "CV Mean":        cv_scores.mean(),
            "CV Std":         cv_scores.std(),
            "Test Accuracy":  test_acc,
            "Test F1":        test_f1,
        })
    except Exception as e:
        results.append({
            "Preprocessor":  prep_name,
            "Model":         model_name,
            "CV Mean":       np.nan,
            "CV Std":        np.nan,
            "Test Accuracy": np.nan,
            "Test F1":       np.nan,
        })

# ── Ranking ───────────────────────────────────────────────────────────────────
ranking = (
    pd.DataFrame(results)
    .dropna()
    .sort_values(["Test Accuracy", "CV Mean", "Test F1"], ascending=False)
    .reset_index(drop=True)
)
ranking.index += 1  # rank starts at 1

pd.set_option("display.max_rows",    None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width",       120)
pd.set_option("display.float_format", "{:.4f}".format)

print("=" * 90)
print(f"{'RANKING — ' + str(len(ranking)) + ' combinations tested':^90}")
print("=" * 90)
print(ranking.to_string())
print("=" * 90)
print("\n🥇 BEST COMBINATION:")
best = ranking.iloc[0]
print(f"   Preprocessor  : {best['Preprocessor']}")
print(f"   Model         : {best['Model']}")
print(f"   Test Accuracy : {best['Test Accuracy']:.4f}")
print(f"   Test F1       : {best['Test F1']:.4f}")
print(f"   CV Mean±Std   : {best['CV Mean']:.4f} ± {best['CV Std']:.4f}")
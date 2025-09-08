import pandas as pd
import joblib, json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_FILE = "random_forest_data.csv"
MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"

def detect_target(df):
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    candidates = ["diagnosis","target","label","result","class","outcome","y","malignant","benign"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name], df
    for c in df.columns:
        if df[c].dtype == "object" and df[c].nunique(dropna=True) == 2:
            return c, df
    return df.columns[-1], df

def encode_labels(y):
    mapping_sets = [
        {"M":1,"B":0,"Malignant":1,"Benign":0,"malignant":1,"benign":0},
        {"Yes":1,"No":0,"yes":1,"no":0,"Y":1,"N":0},
        {"Positive":1,"Negative":0,"positive":1,"negative":0}
    ]
    for m in mapping_sets:
        keys = set(m.keys())
        if set(map(str, y.dropna().unique())).issubset(keys):
            return y.map(m).astype(int), {"mapping": m}
    vals, uniques = pd.factorize(y, sort=True)
    return pd.Series(vals, index=y.index), {"classes": [str(u) for u in uniques]}

def main():
    df = pd.read_csv(DATA_FILE)
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")] + ["id","ID","Id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    target_col, df = detect_target(df)
    feature_df = df.drop(columns=[target_col])
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    X = feature_df[numeric_features].copy()
    y_raw = df[target_col]
    y, label_info = encode_labels(y_raw)
    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                       class_weight="balanced" if y.value_counts().min() / y.value_counts().max() < 0.6 else None))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}
    try:
        if y.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            from sklearn.metrics import roc_auc_score
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception as e:
        metrics["roc_auc_error"] = str(e)
    joblib.dump(pipeline, MODEL_FILE)
    meta = {"features": numeric_features, "target_col": target_col, "label_info": label_info, "metrics": metrics}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved model.joblib and metadata.json")
if __name__ == '__main__':
    main()

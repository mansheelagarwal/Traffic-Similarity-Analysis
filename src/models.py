import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def build_error_model(hourly_df: pd.DataFrame) -> tuple:
    df = hourly_df.copy()
    for col in ["region", "day_type", "time_slice"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    feature_cols = ["hour", "region_enc", "day_type_enc", "time_slice_enc", "flow_pems"]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "flow_pct_error"].abs()
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return model, shap_values, feature_cols, X


def cross_correlation_lag(pems: np.ndarray, sl: np.ndarray, max_lag: int = 3) -> dict:
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(pems, sl)[0, 1]
        elif lag > 0:
            corr = np.corrcoef(pems[lag:], sl[:-lag])[0, 1]
        else:
            corr = np.corrcoef(pems[:lag], sl[-lag:])[0, 1]
        results[lag] = float(corr)
    return results


def compute_umap_embeddings(X: np.ndarray, Y: np.ndarray, n_components: int = 2,
                             random_state: int = 42):
    import umap
    combined = np.vstack([X, Y])
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(combined)
    return embedding[:len(X)], embedding[len(X):]

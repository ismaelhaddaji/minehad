"""
ml_model.py
Machine learning classifier for trade success probability estimation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


ML_FEATURES = [
    'rsi', 'macd', 'macd_hist', 'atr_pct', 'volume_ratio',
    'trend_strength', 'price_vs_vwap', 'bb_pct', 'bb_width',
    'momentum_5', 'momentum_10', 'ema_21_slope', 'ema_50_slope',
    'adx', 'liquidity_sweep', 'structure_break', 'volume_spike',
    'london_kill_zone', 'ny_kill_zone', 'above_vwap',
    'rsi_divergence', 'vol_expansion', 'dxy_trend',
]


def prepare_ml_data(features: pd.DataFrame, forward_periods: int = 10,
                    profit_threshold_pct: float = 0.3) -> tuple:
    """
    Prepare features and labels for ML training.
    Label = 1 if price goes up by threshold% within N periods.
    """
    available_features = [f for f in ML_FEATURES if f in features.columns]
    
    X = features[available_features].copy()
    X = X.fillna(X.mean()).fillna(0)
    
    # Future return as label
    future_return = features['close'].shift(-forward_periods) / features['close'] - 1
    y = (future_return > profit_threshold_pct / 100).astype(int)
    
    # Remove last N rows (no future data)
    X = X.iloc[:-forward_periods]
    y = y.iloc[:-forward_periods]
    
    # Remove NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    return X[mask], y[mask], available_features


def train_ensemble_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train ensemble of ML models."""
    models = {}
    
    # Logistic Regression (baseline)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        random_state=42
    )
    
    model_list = [
        ("logistic_regression", lr_pipeline),
        ("random_forest", rf),
        ("gradient_boosting", gb),
    ]
    
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        model_list.append(("xgboost", xgb))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    cv_scores = {}
    trained_models = {}
    
    for name, model in model_list:
        try:
            if name == "logistic_regression":
                cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                model.fit(X, y)
            else:
                cv = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                model.fit(X_scaled, y)
            
            cv_scores[name] = round(cv.mean(), 3)
            trained_models[name] = model
        except Exception as e:
            cv_scores[name] = 0.5
    
    return {
        "models": trained_models,
        "scaler": scaler,
        "cv_scores": cv_scores,
        "feature_names": list(X.columns),
        "avg_accuracy": round(np.mean(list(cv_scores.values())), 3),
    }


def predict_probability(model_bundle: dict, features: pd.DataFrame) -> dict:
    """Predict trade success probability using ensemble."""
    if not model_bundle or not model_bundle.get("models"):
        return {"probability": 0.5, "confidence": "low", "model_used": "none"}
    
    feature_names = model_bundle.get("feature_names", ML_FEATURES)
    available = [f for f in feature_names if f in features.columns]
    
    latest = features[available].iloc[-1:].copy()
    latest = latest.fillna(0)
    
    # Add missing features as 0
    for f in feature_names:
        if f not in latest.columns:
            latest[f] = 0
    
    latest = latest[feature_names] if all(f in latest.columns for f in feature_names) else latest
    
    scaler = model_bundle.get("scaler")
    if scaler:
        try:
            latest_scaled = scaler.transform(latest)
        except:
            latest_scaled = latest.values
    else:
        latest_scaled = latest.values
    
    probabilities = []
    models = model_bundle.get("models", {})
    
    for name, model in models.items():
        try:
            if name == "logistic_regression":
                prob = model.predict_proba(latest)[0][1]
            else:
                prob = model.predict_proba(latest_scaled)[0][1]
            probabilities.append(prob)
        except:
            probabilities.append(0.5)
    
    if not probabilities:
        return {"probability": 0.5, "confidence": "low", "model_used": "fallback"}
    
    avg_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    
    confidence = "high" if std_prob < 0.05 else ("medium" if std_prob < 0.1 else "low")
    
    return {
        "probability": round(avg_prob, 3),
        "probability_pct": round(avg_prob * 100, 1),
        "std": round(std_prob, 3),
        "confidence": confidence,
        "individual_probs": {k: round(v, 3) for k, v in zip(models.keys(), probabilities)},
        "model_count": len(probabilities),
    }


def get_feature_importance(model_bundle: dict) -> dict:
    """Get feature importances from tree-based models."""
    importances = {}
    models = model_bundle.get("models", {})
    feature_names = model_bundle.get("feature_names", [])
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            fi = dict(zip(feature_names, model.feature_importances_))
            importances[name] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Average importances across models
    if importances:
        all_features = set()
        for fi in importances.values():
            all_features.update(fi.keys())
        
        avg_importance = {}
        for feat in all_features:
            vals = [fi.get(feat, 0) for fi in importances.values()]
            avg_importance[feat] = round(np.mean(vals), 4)
        
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {}


def build_synthetic_model(features: pd.DataFrame) -> dict:
    """Build a quick model from synthetic/historical data when real training data is limited."""
    if len(features) < 50:
        return {}
    
    try:
        X, y, feat_names = prepare_ml_data(features, forward_periods=5, profit_threshold_pct=0.2)
        
        if len(X) < 30 or y.nunique() < 2:
            return {}
        
        return train_ensemble_model(X, y)
    except Exception as e:
        return {}

"""
Local (per-alert) explanation engine using TreeSHAP.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

# Robust path handling
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def _recompute_risk_score(df, iso_forest, scaler, feature_cols):
    """Recreate the risk score column under its training-time name, from the
    saved artifacts — never trust a renamed/copied column from another file."""
    X = df[feature_cols].fillna(0.0)
    raw = iso_forest.decision_function(X)
    risk = scaler.transform(-raw.reshape(-1, 1)).flatten()
    return np.clip(risk, 0, 100)


def _format_reason(feat_name, feat_val, translation_map):
    if feat_name in translation_map:
        try:
            return translation_map[feat_name].format(val=feat_val)
        except (KeyError, ValueError):
            return translation_map[feat_name]

    readable = feat_name.replace("_", " ")
    return f"{readable.capitalize()} at {feat_val:.2f}, deviating from learned baseline"


def generate_local_explanations(
    scored_path=os.path.join(MODEL_DIR, "scored_test_logs.parquet"),
    model_path=os.path.join(MODEL_DIR, "lgbm_classifier.pkl"),
    output_path=os.path.join(MODEL_DIR, "explained_alerts.parquet"),
    alert_budget_pct=0.01,
):
    print("--- [1/4] Loading model, artifacts, and scored test set ---")
    df_scored = pd.read_parquet(scored_path)
    model = joblib.load(model_path)
    iso_forest = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        causal_feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, "model_input_columns.json")) as f:
        model_input_cols = json.load(f)  
    with open(os.path.join(MODEL_DIR, "class_mapping.json")) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    budget_n = max(1, int(len(df_scored) * alert_budget_pct))
    df_alerts = df_scored.iloc[:budget_n].copy()
    print(f"Explaining top {budget_n:,} alerts (of {len(df_scored):,} test events)\n")

    # [2/4] Rebuild the EXACT model input matrix
    print("--- [2/4] Reconstructing model input schema ---")
    risk_col_name = [c for c in model_input_cols if c not in causal_feature_cols]
    assert len(risk_col_name) == 1, (
        "Expected exactly one non-causal column (the risk score) in "
        "model_input_columns.json — schema drift detected, stop and check "
        "train_models.py against this script."
    )
    risk_col_name = risk_col_name[0]

    df_alerts[risk_col_name] = _recompute_risk_score(
        df_alerts, iso_forest, scaler, causal_feature_cols
    )
    X_alerts = df_alerts[model_input_cols].fillna(0.0)
    print(f"Model input matrix shape: {X_alerts.shape} "
          f"(expected {len(model_input_cols)} columns)\n")

    # [3/4] TreeSHAP 
    print("--- [3/4] Running TreeSHAP over the alert budget ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_alerts)

    translation_map = {
        "geo_velocity_kmh": "Physical impossibility: geographic velocity of {val:.1f} km/h",
        "resource_novelty": "First-time access to an unfamiliar endpoint (novelty {val:.2f})",
        "global_resource_rarity": "Access to an organization-wide rare asset (rarity {val:.2f})",
        "device_novelty": "Unrecognized device/OS fingerprint for this entity",
        "ip_fail_velocity_5m": "{val:.0f} failed auth attempts from this IP in 5 minutes",
        "entity_fail_velocity_5m": "{val:.0f} failed auth attempts against this entity in 5 minutes",
        "avg_sequence_surprise": "Highly abnormal command/action sequence for this entity",
        "duration_zscore": "Session length {val:.1f} std deviations from this entity's baseline",
        "session_duration": "Session duration of {val:.1f} min deviates from this entity's baseline",
        "auth_method_risk": "Weak authentication method used (risk level {val:.0f}/3)",
        "ip_novelty": "Login from a previously unseen source IP address",
        "sensitive_access_count_7d": "{val:.0f} rare/sensitive resource accesses in the trailing 7-day window",
        "sequence_anomaly_score": "LSTM autoencoder flagged abnormal command sequence (reconstruction error {val:.2f})",
        risk_col_name: "Stage-1 unsupervised risk score reached {val:.1f}/100",
    }

    print("--- [4/4] Extracting top-2 per-alert reasons ---")
    explanations = []
    for i in range(len(df_alerts)):
        pred_label = df_alerts.iloc[i]["predicted_label"]
        pred_idx = class_to_idx[pred_label]

        row_shaps = (
            shap_values[pred_idx][i]
            if isinstance(shap_values, list)
            else shap_values[i, :, pred_idx]
        )
        ranked = np.argsort(row_shaps)[::-1]

        reasons = []
        for feat_idx in ranked:
            if row_shaps[feat_idx] <= 0:
                break  # remaining features push AWAY from this class; stop
            feat_name = model_input_cols[feat_idx]
            feat_val = X_alerts.iloc[i][feat_name]
            reasons.append(_format_reason(feat_name, feat_val, translation_map))
            if len(reasons) == 2:
                break

        if not reasons:
            reasons.append("Borderline prediction: no single feature dominated the decision")

        explanations.append(" | ".join(reasons))

    df_alerts["soc_explanation"] = explanations
    df_alerts.to_parquet(output_path, index=False)

    print(f"\nSaved explained alerts to: {output_path}")
    print("\nSample explanations (should differ alert-to-alert):")
    for _, row in df_alerts[["predicted_label", "soc_explanation"]].head(5).iterrows():
        print(f"[{row['predicted_label'].upper()}] -> {row['soc_explanation']}")

if __name__ == "__main__":
    generate_local_explanations()
